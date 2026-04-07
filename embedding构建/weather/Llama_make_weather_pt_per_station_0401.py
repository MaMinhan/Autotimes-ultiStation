import os
import re
import argparse
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM


@torch.no_grad()
def embed_texts_llama_original(
    texts: List[str],
    tokenizer,
    model,
    device,
    batch_size: int = 64,
    max_length: int = 192,
) -> torch.Tensor:
    """
    严格对齐原始 Preprocess_Llama.py 的风格：
    1) 不手动补 EOS
    2) tokenizer 得到 input_ids
    3) 用 get_input_embeddings() 转成 input embeddings
    4) 调用 llama.model(inputs_embeds=...)
    5) 取最后一个位置 [:, -1, :]
    """
    model.eval()
    embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)

        inputs_embeds = model.get_input_embeddings()(input_ids)
        text_outputs = model.model(inputs_embeds=inputs_embeds)[0]
        batch_embs = text_outputs[:, -1, :]
        embs.append(batch_embs.detach().cpu())

    return torch.cat(embs, dim=0)


def clean_hourly_temperature_series(
    s: pd.Series,
    min_valid_temp: float = -20.0,
    max_valid_temp: float = 60.0,
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
    if diff < -threshold:
        return "falling"
    return "stable"


def build_temp_window_text(
    temps_window: np.ndarray,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> str:
    start_temp = float(temps_window[0])
    end_temp = float(temps_window[-1])
    mean_temp = float(np.mean(temps_window))
    _min_temp = float(np.min(temps_window))
    _max_temp = float(np.max(temps_window))
    _trend = describe_trend(start_temp, end_temp)

    return (
        f"This is the series from {start_time:%Y-%m-%d %H:%M:%S} "
        f"to {end_time:%Y-%m-%d %H:%M:%S}. "
        f"The temperature has an average of {mean_temp:.1f} degrees Celsius. "
    )


def sanitize_filename(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unknown_station"


def save_station_tensor(out_dir: str, sid: int, station_tensor: torch.Tensor) -> str:
    filename = f"station_{sid}.pt"
    out_path = os.path.join(out_dir, filename)
    torch.save(station_tensor, out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--power_csv", type=str, required=True,
                    help="Electricity CSV, must include datetime and station_id")
    ap.add_argument("--weather_csv", type=str, required=True,
                    help="Hourly weather CSV, must include date, station_id, temperature_2m_mean")
    ap.add_argument("--llm_ckp_dir", type=str, required=True,
                    help="HF model dir, e.g. /root/autodl-tmp/hf_models/llama")
    ap.add_argument("--out_weather_dir", type=str, required=True,
                    help="Output directory. Each station will be saved as one pt file with shape [T, D].")

    ap.add_argument("--token_len", type=int, required=True,
                    help="Token length used by the forecasting model")

    ap.add_argument("--stations_limit", type=int, default=0,
                    help="For debugging: only keep first N stations. 0 = all")
    ap.add_argument("--days_limit", type=int, default=0,
                    help="For debugging: only keep first N days per station after alignment. 0 = all")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])

    ap.add_argument("--min_valid_temp", type=float, default=-20.0)
    ap.add_argument("--max_valid_temp", type=float, default=60.0)
    ap.add_argument("--save_meta", action="store_true",
                    help="Whether to additionally save meta info such as station_ids and global datetime axis.")
    ap.add_argument("--start_station_id", type=int, default=None)
    args = ap.parse_args()

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
    N = len(station_ids)

    print(f"[POWER] N={N}, T={T}")
    print(f"[POWER] dt range: {dt.iloc[0]} -> {dt.iloc[-1]}")
    print(f"[CONFIG] token_len={args.token_len}")

    print("[2/5] Reading weather csv...")
    dfw = pd.read_csv(
        args.weather_csv,
        usecols=["date", "station_id", "temperature_2m_mean"],
    )
    dfw["date"] = pd.to_datetime(dfw["date"], errors="coerce")
    dfw = dfw.dropna(subset=["date", "station_id", "temperature_2m_mean"])
    dfw["station_id"] = dfw["station_id"].astype(int)
    dfw["temperature_2m_mean"] = pd.to_numeric(dfw["temperature_2m_mean"], errors="coerce")
    dfw = dfw.groupby(["station_id", "date"], as_index=False).mean(numeric_only=True)
    dfw = dfw[dfw["station_id"].isin(station_ids)].copy()

    print(f"[WEATHER] rows(after groupby/filter)={len(dfw)}")
    print(f"[WEATHER] station count={dfw['station_id'].nunique()}")

    print("[3/5] Loading tokenizer/model...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = LlamaTokenizer.from_pretrained(args.llm_ckp_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(
        args.llm_ckp_dir,
        torch_dtype=torch.float16 if args.dtype == "float16" else torch.float32,
        device_map=None,
    ).to(device)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    D = model.config.hidden_size
    out_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    os.makedirs(args.out_weather_dir, exist_ok=True)

    missing_text = "Temperature data is unavailable for this station and time window."
    missing_emb = embed_texts_llama_original(
        texts=[missing_text],
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=1,
        max_length=args.max_length,
    )[0]
    missing_emb = missing_emb.half() if out_dtype == torch.float16 else missing_emb.float()

    print(f"[EMBED] hidden_size={D}, dtype={out_dtype}")
    print(f"[SAVE] per-station directory={args.out_weather_dir}")

    print("[4/5] Generating and saving per-station embeddings...")
    dt_index = pd.DatetimeIndex(dt)
    global_dates = pd.Series(dt).dt.normalize()

    saved_files = []

    for sid in station_ids:
        if args.start_station_id is not None and sid < args.start_station_id:
            continue
        g = dfw[dfw["station_id"] == sid].copy()
        station_pt = missing_emb.unsqueeze(0).repeat(T, 1).clone()

        if len(g) == 0:
            out_path = save_station_tensor(args.out_weather_dir, sid, station_pt)
            saved_files.append((sid, out_path))
            print(f"[WARN] sid={sid} has no matched weather rows, saved missing embedding only -> {out_path}")
            continue

        g = g.sort_values("date").set_index("date")
        s = g["temperature_2m_mean"]

        raw_zero_cnt = int((s == 0).sum())
        raw_nan_cnt = int(s.isna().sum())

        s_hourly = clean_hourly_temperature_series(
            s,
            min_valid_temp=args.min_valid_temp,
            max_valid_temp=args.max_valid_temp,
        )
        cleaned_nan_cnt = int(s_hourly.isna().sum())

        s_15 = hourly_to_15min_temperature(s_hourly)
        s_aligned = s_15.reindex(dt_index)
        s_aligned = s_aligned.interpolate(method="time").ffill().bfill()

        if args.days_limit > 0:
            keep_dates = global_dates.drop_duplicates().iloc[:args.days_limit]
            keep_mask = global_dates.isin(set(keep_dates))
            valid_tidx = np.where(keep_mask.values)[0]
        else:
            valid_tidx = np.arange(T)

        candidate_tidx = [t for t in valid_tidx if t + args.token_len <= T]
        if len(candidate_tidx) == 0:
            out_path = save_station_tensor(args.out_weather_dir, sid, station_pt)
            saved_files.append((sid, out_path))
            print(f"[WARN] sid={sid} has no valid token windows, saved missing embedding only -> {out_path}")
            continue

        texts = []
        for t in candidate_tidx:
            temps_window = s_aligned.iloc[t:t + args.token_len].to_numpy(dtype=np.float32)
            start_time = dt.iloc[t]
            end_time = dt.iloc[t + args.token_len - 1]
            texts.append(build_temp_window_text(temps_window, start_time, end_time))

        embs = embed_texts_llama_original(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        embs = embs.half() if out_dtype == torch.float16 else embs.float()

        for k, t in enumerate(candidate_tidx):
            station_pt[t, :] = embs[k]

        out_path = save_station_tensor(args.out_weather_dir, sid, station_pt)
        saved_files.append((sid, out_path))

        nz_ratio = float((station_pt.abs().sum(dim=-1) != 0).float().mean().item())
        print(
            f"[STATION] sid={sid}, valid_windows={len(candidate_tidx)}, "
            f"raw_zero_cnt={raw_zero_cnt}, raw_nan_cnt={raw_nan_cnt}, "
            f"cleaned_nan_cnt={cleaned_nan_cnt}, "
            f"temp_mean={float(s_aligned.mean()):.3f}, "
            f"temp_min={float(s_aligned.min()):.3f}, temp_max={float(s_aligned.max()):.3f}, "
            f"saved={out_path}, nz_ratio={nz_ratio:.6f}"
        )

    print("[5/5] Finishing...")
    print(f"[OK] saved {len(saved_files)} station pt files into: {args.out_weather_dir}")

    if args.save_meta:
        meta = {
            "station_ids": station_ids,
            "dt": [x.strftime("%Y-%m-%d %H:%M:%S") for x in dt],
            "token_len": int(args.token_len),
            "hidden_size": int(D),
            "dtype": str(out_dtype),
        }
        meta_path = os.path.join(args.out_weather_dir, "meta.pt")
        torch.save(meta, meta_path)
        print(f"[OK] saved meta -> {meta_path}")


if __name__ == "__main__":
    main()
'''python /root/autotimes/embedding构建/weather/Llama_make_weather_pt_per_station_0401.py \
  --power_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --weather_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/weather_hourly_20210501_20240430.csv \
  --llm_ckp_dir /root/autodl-tmp/hf_models/llama \
  --out_weather_dir /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/0401_per_station \
  --token_len 96 \
  --batch_size 9\
  --max_length 192 \
  --device cuda \
  --dtype float16'''