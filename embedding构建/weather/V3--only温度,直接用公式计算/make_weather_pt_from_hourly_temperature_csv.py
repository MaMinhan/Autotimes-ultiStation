import os
import argparse
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel



def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


@torch.no_grad()
def embed_texts(
    texts: List[str],
    tokenizer,
    model,
    device,
    batch_size: int = 64,
    max_length: int = 128
) -> torch.Tensor:
    model.eval()
    embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        e = mean_pool(out.last_hidden_state, enc["attention_mask"])
        embs.append(e.detach().cpu())

    return torch.cat(embs, dim=0)


def build_temp_text(temp_c: float) -> str:
    return f"Air temperature is {temp_c:.1f} degrees Celsius."


def clean_hourly_temperature_series(
    s: pd.Series,
    min_valid_temp: float = -20.0,
    max_valid_temp: float = 60.0
) -> pd.Series:
    """
    固定规则：
    1. 0 保留（不特殊处理）
    2. NaN / inf / -inf -> 缺失
    3. 超出 [min_valid_temp, max_valid_temp] -> 缺失
    4. 在 hourly 层面按时间插值
    5. 首尾 ffill / bfill
    """
    s = s.astype(float).copy()

    # inf / -inf -> nan
    s[~np.isfinite(s)] = np.nan

    # 只处理明显异常值，0 不动
    s[(s < min_valid_temp) | (s > max_valid_temp)] = np.nan

    # hourly层面插值
    s = s.interpolate(method="time")
    s = s.ffill().bfill()

    return s


def hourly_to_15min_temperature(s_hourly: pd.Series) -> pd.Series:
    """
    固定逻辑：
    hourly -> 15min 使用时间线性插值
    """
    s_15 = s_hourly.resample("15min").interpolate(method="time")
    s_15 = s_15.ffill().bfill()
    return s_15


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

    ap.add_argument("--stations_limit", type=int, default=0,
                    help="For debugging: only keep first N stations. 0 = all")
    ap.add_argument("--days_limit", type=int, default=0,
                    help="For debugging: only keep first N days per station after alignment. 0 = all")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])

    # 温度合法范围，可按需要改
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
    model = AutoModel.from_pretrained(args.llm_ckp_dir).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    D = model.config.hidden_size
    out_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    weather_pt = torch.zeros((N, T, D), dtype=out_dtype)

    missing_text = "Temperature data is unavailable for this station."
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
    #    clean hourly -> hourly interpolate -> 15min interpolate
    #    -> align to global dt -> text -> embedding
    # ------------------------------------------------------------
    print("[4/5] Generating embeddings...")

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

        # 只留温度列
        s = g["temperature_2m_mean"]

        raw_zero_cnt = int((s == 0).sum())
        raw_nan_cnt = int(s.isna().sum())

        # ---- 固定清洗逻辑：0保留，只处理NaN/inf/异常值 ----
        s_hourly = clean_hourly_temperature_series(
            s,
            min_valid_temp=args.min_valid_temp,
            max_valid_temp=args.max_valid_temp
        )

        cleaned_nan_cnt = int(s_hourly.isna().sum())

        # ---- 固定逻辑：hourly -> 15min 线性插值 ----
        s_15 = hourly_to_15min_temperature(s_hourly)

        # ---- 对齐到 power 的全局时间轴 ----
        s_aligned = s_15.reindex(dt_index)
        s_aligned = s_aligned.interpolate(method="time").ffill().bfill()

        # 如果调试，只保留前 N 天对应的时间点，其余不写入
        if args.days_limit > 0:
            keep_dates = global_dates.drop_duplicates().iloc[:args.days_limit]
            keep_mask = global_dates.isin(set(keep_dates))
            tidx = np.where(keep_mask.values)[0]
        else:
            tidx = np.arange(T)

        temps = s_aligned.iloc[tidx].to_numpy(dtype=np.float32)
        times = dt.iloc[tidx].tolist()

        texts = [
            build_temp_text(float(temp))
            for temp in temps
        ]

        embs = embed_texts(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )  # [len(tidx), D]

        embs = embs.half() if out_dtype == torch.float16 else embs.float()
        weather_pt[sid_idx, tidx, :] = embs

        print(
            f"[STATION] sid={sid}, written_slots={len(tidx)}, "
            f"raw_zero_cnt={raw_zero_cnt}, raw_nan_cnt={raw_nan_cnt}, "
            f"cleaned_nan_cnt={cleaned_nan_cnt}, "
            f"temp_mean={temps.mean():.3f}, temp_min={temps.min():.3f}, temp_max={temps.max():.3f}"
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