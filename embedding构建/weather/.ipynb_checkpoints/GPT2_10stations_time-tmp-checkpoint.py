import os
import argparse
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


@torch.no_grad()
def embed_texts(
    texts: List[str],
    tokenizer,
    model,
    device,
    batch_size: int = 64,
    max_length: int = 192
) -> torch.Tensor:
    model.eval()
    embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # 显式加 EOS
        batch = [t + " " + tokenizer.eos_token for t in batch]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        last_hidden = out.last_hidden_state  # [B, L, D]

        eos_idx = enc["attention_mask"].sum(dim=1) - 1
        batch_idx = torch.arange(last_hidden.size(0), device=device)
        e = last_hidden[batch_idx, eos_idx, :]  # [B, D]

        embs.append(e.detach().cpu())

    return torch.cat(embs, dim=0)


def clean_hourly_temperature_series(
    s: pd.Series,
    min_valid_temp: float = -20.0,
    max_valid_temp: float = 60.0
) -> pd.Series:
    s = s.astype(float).copy()
    s[~np.isfinite(s)] = np.nan
    s[(s < min_valid_temp) | (s > max_valid_temp)] = np.nan
    s = s.interpolate(method="time")
    s = s.ffill().bfill()
    return s


def hourly_to_15min_temperature(s_hourly: pd.Series) -> pd.Series:
    s_15 = s_hourly.resample("15min").interpolate(method="time")
    s_15 = s_15.ffill().bfill()
    return s_15


def describe_trend(start_temp: float, end_temp: float, threshold: float = 0.5) -> str:
    diff = end_temp - start_temp
    if diff > threshold:
        return "rising"
    elif diff < -threshold:
        return "falling"
    else:
        return "stable"


def build_temp_window_text(
    temps_window: np.ndarray,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp
) -> str:
    """
    为一个 token_len 时间窗口构造天气文本。
    语义上表达的是：
    这段 series 对应时间窗口内的辅助温度信息。
    """
    start_temp = float(temps_window[0])
    end_temp = float(temps_window[-1])
    mean_temp = float(np.mean(temps_window))
    min_temp = float(np.min(temps_window))
    max_temp = float(np.max(temps_window))
    trend = describe_trend(start_temp, end_temp)

    return (
        f"This is the series from {start_time:%Y-%m-%d %H:%M:%S} "
        f"to {end_time:%Y-%m-%d %H:%M:%S}. "
        f"The temperature has an average of {mean_temp:.1f} degrees Celsius. "
    )



def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--power_csv", type=str, required=True,
                    help="Electricity CSV, must include datetime and station_id")
    ap.add_argument("--weather_csv", type=str, required=True,
                    help="Hourly weather CSV, must include date, station_id, temperature_2m_mean")
    ap.add_argument("--llm_ckp_dir", type=str, required=True,
                    help="HF model dir, e.g. /root/autodl-tmp/hf_models/gpt2")
    ap.add_argument("--out_weather_pt", type=str, required=True,
                    help="Output weather.pt path, shape [N, T, D]")

    # 新增：token_len
    ap.add_argument("--token_len", type=int, required=True,
                    help="Token length used by the forecasting model")

    ap.add_argument("--stations_limit", type=int, default=0,
                    help="For debugging: only keep first N stations. 0 = all")
    ap.add_argument("--days_limit", type=int, default=0,
                    help="For debugging: only keep first N days per station after alignment. 0 = all")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])

    ap.add_argument("--min_valid_temp", type=float, default=-20.0)
    ap.add_argument("--max_valid_temp", type=float, default=60.0)

    args = ap.parse_args()

    # ------------------------------------------------------------
    # 1) Read power csv -> global 15min time axis + station list
    # ------------------------------------------------------------
    print("[1/5] Reading power csv...")
    dfp = pd.read_csv(args.power_csv, usecols=["datetime", "station_id"])
    dfp["datetime"] = pd.to_datetime(dfp["datetime"], errors="coerce")
    dfp = dfp.dropna(subset=["datetime", "station_id"])
    dfp["station_id"] = dfp["station_id"].astype(int)

    dt = dfp["datetime"].drop_duplicates().sort_values().reset_index(drop=True)
    T = len(dt)

    station_ids = sorted(dfp["station_id"].unique().tolist())
    if args.stations_limit > 0:
        station_ids = station_ids[:args.stations_limit]

    sid2idx = {sid: i for i, sid in enumerate(station_ids)}
    N = len(station_ids)

    print(f"[POWER] N={N}, T={T}")
    print(f"[POWER] dt range: {dt.iloc[0]} -> {dt.iloc[-1]}")
    print(f"[CONFIG] token_len={args.token_len}")

    # ------------------------------------------------------------
    # 2) Read weather csv (hourly), only use temperature_2m_mean
    # ------------------------------------------------------------
    print("[2/5] Reading weather csv...")
    dfw = pd.read_csv(
        args.weather_csv,
        usecols=["date", "station_id", "temperature_2m_mean"]
    )

    dfw["date"] = pd.to_datetime(dfw["date"], errors="coerce")
    dfw = dfw.dropna(subset=["date", "station_id", "temperature_2m_mean"])
    dfw["station_id"] = dfw["station_id"].astype(int)
    dfw["temperature_2m_mean"] = pd.to_numeric(dfw["temperature_2m_mean"], errors="coerce")

    # 同一站点同一时刻若有重复，取均值
    dfw = dfw.groupby(["station_id", "date"], as_index=False).mean(numeric_only=True)

    # 只保留 power 里存在的 station
    dfw = dfw[dfw["station_id"].isin(station_ids)].copy()

    print(f"[WEATHER] rows(after groupby/filter)={len(dfw)}")
    print(f"[WEATHER] station count={dfw['station_id'].nunique()}")

    # ------------------------------------------------------------
    # 3) Load embedding model
    # ------------------------------------------------------------
    print("[3/5] Loading tokenizer/model...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.llm_ckp_dir)
    if tokenizer.eos_token is None:
        raise ValueError("GPT2 tokenizer has no eos_token, please check llm_ckp_dir.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2Model.from_pretrained(args.llm_ckp_dir).to(device)

    D = model.config.hidden_size
    out_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    weather_pt = torch.zeros((N, T, D), dtype=out_dtype)

    missing_text = "Temperature data is unavailable for this station and time window."
    missing_emb = embed_texts(
        texts=[missing_text],
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=1,
        max_length=args.max_length,
    )[0]
    missing_emb = missing_emb.half() if out_dtype == torch.float16 else missing_emb.float()

    print(f"[EMBED] hidden_size={D}, dtype={out_dtype}")
    print(f"[ALLOC] weather_pt shape={tuple(weather_pt.shape)}")

    # ------------------------------------------------------------
    # 4) For each station:
    #    clean hourly -> 15min interpolate -> align to global dt
    #    -> token-window text -> embedding
    # ------------------------------------------------------------
    print("[4/5] Generating token-window embeddings...")

    dt_index = pd.DatetimeIndex(dt)
    global_dates = pd.Series(dt).dt.normalize()

    for sid in station_ids:
        sid_idx = sid2idx[sid]

        g = dfw[dfw["station_id"] == sid].copy()
        if len(g) == 0:
            print(f"[WARN] sid={sid} has no matched weather rows, fill with missing embedding.")
            weather_pt[sid_idx, :, :] = missing_emb.unsqueeze(0).repeat(T, 1)
            continue

        g = g.sort_values("date")
        g = g.set_index("date")

        s = g["temperature_2m_mean"]

        raw_zero_cnt = int((s == 0).sum())
        raw_nan_cnt = int(s.isna().sum())

        s_hourly = clean_hourly_temperature_series(
            s,
            min_valid_temp=args.min_valid_temp,
            max_valid_temp=args.max_valid_temp
        )
        cleaned_nan_cnt = int(s_hourly.isna().sum())

        s_15 = hourly_to_15min_temperature(s_hourly)

        s_aligned = s_15.reindex(dt_index)
        s_aligned = s_aligned.interpolate(method="time").ffill().bfill()

        # debug only: keep first N days
        if args.days_limit > 0:
            keep_dates = global_dates.drop_duplicates().iloc[:args.days_limit]
            keep_mask = global_dates.isin(set(keep_dates))
            valid_tidx = np.where(keep_mask.values)[0]
        else:
            valid_tidx = np.arange(T)

        # 只对能形成完整 token window 的起点生成文本
        candidate_tidx = [t for t in valid_tidx if t + args.token_len <= T]

        if len(candidate_tidx) == 0:
            print(f"[WARN] sid={sid} has no valid token windows, fill with missing embedding.")
            weather_pt[sid_idx, :, :] = missing_emb.unsqueeze(0).repeat(T, 1)
            continue

        texts = []
        for t in candidate_tidx:
            temps_window = s_aligned.iloc[t:t + args.token_len].to_numpy(dtype=np.float32)
            start_time = dt.iloc[t]
            end_time = dt.iloc[t + args.token_len - 1]

            text = build_temp_window_text(
                temps_window=temps_window,
                start_time=start_time,
                end_time=end_time
            )
            texts.append(text)

        embs = embed_texts(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )  # [len(candidate_tidx), D]

        embs = embs.half() if out_dtype == torch.float16 else embs.float()

        # 先全部填 missing，再覆盖有效位置
        weather_pt[sid_idx, :, :] = missing_emb.unsqueeze(0).repeat(T, 1)
        for k, t in enumerate(candidate_tidx):
            weather_pt[sid_idx, t, :] = embs[k]

        # 对最后不足一个 token_len 的尾巴，保留 missing_emb
        print(
            f"[STATION] sid={sid}, valid_windows={len(candidate_tidx)}, "
            f"raw_zero_cnt={raw_zero_cnt}, raw_nan_cnt={raw_nan_cnt}, "
            f"cleaned_nan_cnt={cleaned_nan_cnt}, "
            f"temp_mean={float(s_aligned.mean()):.3f}, "
            f"temp_min={float(s_aligned.min()):.3f}, temp_max={float(s_aligned.max()):.3f}"
        )

    # ------------------------------------------------------------
    # 5) Save
    # ------------------------------------------------------------
    print("[5/5] Saving...")
    out_dir = os.path.dirname(args.out_weather_pt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(weather_pt, args.out_weather_pt)

    nz_ratio = float((weather_pt.abs().sum(dim=-1) != 0).float().mean().item())
    print(f"[OK] saved weather.pt -> {args.out_weather_pt}")
    print(f"[OK] shape={tuple(weather_pt.shape)}, dtype={weather_pt.dtype}")
    print(f"[STAT] nonzero ratio={nz_ratio:.6f}")

    nz = (weather_pt.abs().sum(dim=-1) != 0).nonzero(as_tuple=False)
    if nz.numel() > 0:
        i, t = nz[0].tolist()
        print(f"[SAMPLE] first nonzero at sid_idx={i}, tidx={t}, vec_norm={weather_pt[i, t].float().norm().item():.6f}")


if __name__ == "__main__":
    main()
'''python /root/autotimes/embedding构建/weather/GPT2_weather_time_0406.py \
  --power_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --weather_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/weather_hourly_20210501_20240430.csv \
  --llm_ckp_dir /root/autodl-tmp/hf_models/gpt2 \
  --out_weather_pt /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/0406_time+temp_token96_gpt2.pt \
  --token_len 96 \
  --batch_size 64 \
  --max_length 192 \
  --device cuda \
  --dtype float16'''