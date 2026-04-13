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


def clean_hourly_series(
    s: pd.Series,
    min_valid_value: float,
    max_valid_value: float
) -> pd.Series:
    s = s.astype(float).copy()
    s[~np.isfinite(s)] = np.nan
    s[(s < min_valid_value) | (s > max_valid_value)] = np.nan
    s = s.interpolate(method="time")
    s = s.ffill().bfill()
    return s


def hourly_to_15min_series(s_hourly: pd.Series) -> pd.Series:
    s_15 = s_hourly.resample("15min").interpolate(method="time")
    s_15 = s_15.ffill().bfill()
    return s_15


def describe_trend(start_val: float, end_val: float, threshold: float) -> str:
    diff = end_val - start_val
    if diff > threshold:
        return "rising"
    elif diff < -threshold:
        return "falling"
    else:
        return "stable"


def summarize_window(values_window: np.ndarray, trend_threshold: float):
    start_val = float(values_window[0])
    end_val = float(values_window[-1])
    mean_val = float(np.mean(values_window))
    min_val = float(np.min(values_window))
    max_val = float(np.max(values_window))
    trend = describe_trend(start_val, end_val, threshold=trend_threshold)
    return {
        "start": start_val,
        "end": end_val,
        "mean": mean_val,
        "min": min_val,
        "max": max_val,
        "trend": trend,
    }


def describe_temperature_level(mean_temp: float) -> str:
    if mean_temp < 10:
        return "cold"
    elif mean_temp < 18:
        return "cool"
    elif mean_temp < 26:
        return "mild"
    elif mean_temp < 32:
        return "warm"
    else:
        return "very hot"


def describe_humidity_level(mean_humidity: float) -> str:
    if mean_humidity < 30:
        return "dry"
    elif mean_humidity < 60:
        return "comfortable"
    elif mean_humidity < 80:
        return "humid"
    else:
        return "very humid"


def describe_wind_level(mean_wind: float) -> str:
    if mean_wind < 10:
        return "calm"
    elif mean_wind < 20:
        return "breezy"
    elif mean_wind < 35:
        return "windy"
    else:
        return "very windy"


def describe_generic_level(feature_name: str, mean_val: float) -> str:
    fname = feature_name.strip().lower()
    if fname == "temperature":
        return describe_temperature_level(mean_val)
    if fname == "relative humidity":
        return describe_humidity_level(mean_val)
    if fname == "wind speed":
        return describe_wind_level(mean_val)
    return f"{feature_name} at {mean_val:.1f}"


def describe_temperature_trend(start_val: float, end_val: float, threshold: float = 0.8) -> str:
    diff = end_val - start_val
    if diff > threshold:
        return "getting hotter"
    elif diff < -threshold:
        return "getting cooler"
    else:
        return "remaining stable"


def describe_generic_trend(feature_name: str, start_val: float, end_val: float, threshold: float) -> str:
    fname = feature_name.strip().lower()
    if fname == "temperature":
        return describe_temperature_trend(start_val, end_val, threshold)
    diff = end_val - start_val
    if diff > threshold:
        return f"increasing in {feature_name}"
    elif diff < -threshold:
        return f"decreasing in {feature_name}"
    else:
        return f"stable in {feature_name}"


def build_single_window_text(
    values_window: np.ndarray,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    feature_name: str,
    feature_unit: str,
    trend_threshold: float,
    text_style: str = "simple",
    prompt_mode: str = "semantic",
) -> str:
    s = summarize_window(values_window, trend_threshold)

    def with_unit(v: float, unit: str) -> str:
        return f"{v:.1f} {unit}" if unit else f"{v:.1f}"

    if prompt_mode == "numeric":
        if text_style == "simple":
            return (
                f"This is the series from {start_time:%Y-%m-%d %H:%M:%S} "
                f"to {end_time:%Y-%m-%d %H:%M:%S}. "
                f"The {feature_name} has an average of {with_unit(s['mean'], feature_unit)}. "
            )
        elif text_style == "detailed":
            return (
                f"This is the series from {start_time:%Y-%m-%d %H:%M:%S} "
                f"to {end_time:%Y-%m-%d %H:%M:%S}. "
                f"The {feature_name} has an average of {with_unit(s['mean'], feature_unit)}, "
                f"a minimum of {with_unit(s['min'], feature_unit)}, "
                f"and a maximum of {with_unit(s['max'], feature_unit)}. "
                f"It is {s['trend']} during this period. "
            )
        else:
            raise ValueError(f"Unknown text_style: {text_style}")

    level_desc = describe_generic_level(feature_name, s["mean"])
    trend_desc = describe_generic_trend(feature_name, s["start"], s["end"], trend_threshold)

    if text_style == "simple":
        if feature_name.strip().lower() == "temperature":
            return (
                f"This is the series from {start_time:%Y-%m-%d %H:%M:%S} "
                f"to {end_time:%Y-%m-%d %H:%M:%S}. "
                f"The weather is {level_desc}, {trend_desc}. "
            )
        else:
            return (
                f"This is the series from {start_time:%Y-%m-%d %H:%M:%S} "
                f"to {end_time:%Y-%m-%d %H:%M:%S}. "
                f"The {feature_name} is {level_desc}, {trend_desc}. "
            )
    elif text_style == "detailed":
        if feature_name.strip().lower() == "temperature":
            return (
                f"This is the series from {start_time:%Y-%m-%d %H:%M:%S} "
                f"to {end_time:%Y-%m-%d %H:%M:%S}. "
                f"The weather is {level_desc}, {trend_desc}. "
            )
        else:
            return (
                f"This is the series from {start_time:%Y-%m-%d %H:%M:%S} "
                f"to {end_time:%Y-%m-%d %H:%M:%S}. "
                f"The {feature_name} is {level_desc}, {trend_desc}. "
            )
    else:
        raise ValueError(f"Unknown text_style: {text_style}")


def build_joint_window_text(
    values1_window: np.ndarray,
    values2_window: np.ndarray,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    feature1_name: str,
    feature1_unit: str,
    feature2_name: str,
    feature2_unit: str,
    feature1_trend_threshold: float,
    feature2_trend_threshold: float,
    text_style: str = "simple",
    prompt_mode: str = "semantic",
) -> str:
    s1 = summarize_window(values1_window, feature1_trend_threshold)
    s2 = summarize_window(values2_window, feature2_trend_threshold)

    f1 = feature1_name.strip().lower()
    f2 = feature2_name.strip().lower()

    def with_unit(v: float, unit: str) -> str:
        return f"{v:.1f} {unit}" if unit else f"{v:.1f}"

    if prompt_mode == "numeric":
        if text_style == "simple":
            return (
                f"This is the series from {start_time:%Y-%m-%d %H:%M:%S} "
                f"to {end_time:%Y-%m-%d %H:%M:%S}. "
                f"The {feature1_name} has an average of {with_unit(s1['mean'], feature1_unit)}. "
                f"The {feature2_name} has an average of {with_unit(s2['mean'], feature2_unit)}. "
            )
        elif text_style == "detailed":
            return (
                f"This is the series from {start_time:%Y-%m-%d %H:%M:%S} "
                f"to {end_time:%Y-%m-%d %H:%M:%S}. "
                f"The {feature1_name} has an average of {with_unit(s1['mean'], feature1_unit)}, "
                f"a minimum of {with_unit(s1['min'], feature1_unit)}, "
                f"and a maximum of {with_unit(s1['max'], feature1_unit)}. "
                f"It is {s1['trend']} during this period. "
                f"The {feature2_name} has an average of {with_unit(s2['mean'], feature2_unit)}, "
                f"a minimum of {with_unit(s2['min'], feature2_unit)}, "
                f"and a maximum of {with_unit(s2['max'], feature2_unit)}. "
                f"It is {s2['trend']} during this period. "
            )
        else:
            raise ValueError(f"Unknown text_style: {text_style}")

    if f1 == "temperature" and f2 == "relative humidity":
        temp_desc = describe_temperature_level(s1["mean"])
        humid_desc = describe_humidity_level(s2["mean"])
        trend_desc = describe_temperature_trend(s1["start"], s1["end"], feature1_trend_threshold)

        if text_style == "simple":
            return (
                f"This is the series from {start_time:%Y-%m-%d %H:%M:%S} "
                f"to {end_time:%Y-%m-%d %H:%M:%S}. "
                f"The weather is {temp_desc} and {humid_desc}, {trend_desc}. "
            )
        elif text_style == "detailed":
            return (
                f"This is the series from {start_time:%Y-%m-%d %H:%M:%S} "
                f"to {end_time:%Y-%m-%d %H:%M:%S}. "
                f"The weather is {temp_desc} and {humid_desc}. "
                f"The temperature is {trend_desc} during this period. "
            )
        else:
            raise ValueError(f"Unknown text_style: {text_style}")

    if f1 == "temperature" and f2 == "wind speed":
        temp_desc = describe_temperature_level(s1["mean"])
        wind_desc = describe_wind_level(s2["mean"])
        trend_desc = describe_temperature_trend(s1["start"], s1["end"], feature1_trend_threshold)

        if text_style == "simple":
            return (
                f"This is the series from {start_time:%Y-%m-%d %H:%M:%S} "
                f"to {end_time:%Y-%m-%d %H:%M:%S}. "
                f"The weather is {temp_desc} and {wind_desc}, {trend_desc}. "
            )
        elif text_style == "detailed":
            return (
                f"This is the series from {start_time:%Y-%m-%d %H:%M:%S} "
                f"to {end_time:%Y-%m-%d %H:%M:%S}. "
                f"The weather is {temp_desc} and {wind_desc}. "
                f"The temperature is {trend_desc} during this period. "
            )
        else:
            raise ValueError(f"Unknown text_style: {text_style}")

    level1 = describe_generic_level(feature1_name, s1["mean"])
    level2 = describe_generic_level(feature2_name, s2["mean"])
    trend1 = describe_generic_trend(feature1_name, s1["start"], s1["end"], feature1_trend_threshold)

    if text_style == "simple":
        return (
            f"This is the series from {start_time:%Y-%m-%d %H:%M:%S} "
            f"to {end_time:%Y-%m-%d %H:%M:%S}. "
            f"The {feature1_name} is {level1}. "
            f"The {feature2_name} is {level2}. "
            f"The {feature1_name} is {trend1}. "
        )
    elif text_style == "detailed":
        return (
            f"This is the series from {start_time:%Y-%m-%d %H:%M:%S} "
            f"to {end_time:%Y-%m-%d %H:%M:%S}. "
            f"The {feature1_name} is {level1}, and the {feature2_name} is {level2}. "
            f"The {feature1_name} is {trend1} during this period. "
        )
    else:
        raise ValueError(f"Unknown text_style: {text_style}")
def main():
    ap = argparse.ArgumentParser()

    # -------------------------------
    # 输入输出
    # -------------------------------
    ap.add_argument("--power_csv", type=str, required=True,
                    help="Electricity CSV, must include datetime and station_id")
    ap.add_argument(
        "--weather_csv",
        type=str,
        required=True,
        help="Hourly weather CSV, must include date, station_id, feature1_col, and optionally feature2_col"
    )   
    ap.add_argument("--llm_ckp_dir", type=str, required=True,
                    help="HF model dir, e.g. /root/autodl-tmp/hf_models/gpt2")
    ap.add_argument("--out_feature_pt", type=str, required=True,
                    help="Output joint feature .pt path, shape [N, T, D]")
    ap.add_argument("--prompt_mode", type=str, default="semantic", choices=["semantic", "numeric"])
    # -------------------------------
    # 特征1
    # -------------------------------
    ap.add_argument("--feature1_col", type=str, required=True)
    ap.add_argument("--feature1_name", type=str, required=True)
    ap.add_argument("--feature1_unit", type=str, default="")
    ap.add_argument("--feature1_min", type=float, required=True)
    ap.add_argument("--feature1_max", type=float, required=True)
    ap.add_argument("--feature1_trend_threshold", type=float, default=0.5)

    # -------------------------------
    # 特征2
    # -------------------------------
    # 特征2：改成可选
    ap.add_argument("--feature2_col", type=str, default="")
    ap.add_argument("--feature2_name", type=str, default="")
    ap.add_argument("--feature2_unit", type=str, default="")
    ap.add_argument("--feature2_min", type=float, default=None)
    ap.add_argument("--feature2_max", type=float, default=None)
    ap.add_argument("--feature2_trend_threshold", type=float, default=0.5)
    # -------------------------------
    # Prompt 配置
    # -------------------------------
    ap.add_argument("--missing_text", type=str, default="")
    ap.add_argument("--text_style", type=str, default="simple", choices=["simple", "detailed"])

    # -------------------------------
    # 时序参数
    # -------------------------------
    ap.add_argument("--token_len", type=int, required=True)

    # -------------------------------
    # 调试参数
    # -------------------------------
    ap.add_argument("--stations_limit", type=int, default=0)
    ap.add_argument("--days_limit", type=int, default=0)

    # -------------------------------
    # embedding 参数
    # -------------------------------
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])

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
    # 2) Read weather csv
    # ------------------------------------------------------------
    print("[2/5] Reading weather csv...")
    use_feature2 = bool(args.feature2_col.strip())

    required_cols = ["date", "station_id", args.feature1_col]
    if use_feature2:
        if not args.feature2_name.strip():
            raise ValueError("When feature2_col is provided, feature2_name must also be provided.")
        if args.feature2_min is None or args.feature2_max is None:
            raise ValueError("When feature2_col is provided, feature2_min and feature2_max must also be provided.")
        required_cols.append(args.feature2_col)

    dfw = pd.read_csv(args.weather_csv, usecols=required_cols)

    dfw["date"] = pd.to_datetime(dfw["date"], errors="coerce")
    dfw = dfw.dropna(subset=["date", "station_id"])
    dfw["station_id"] = dfw["station_id"].astype(int)
    dfw[args.feature1_col] = pd.to_numeric(dfw[args.feature1_col], errors="coerce")

    if use_feature2:
        dfw[args.feature2_col] = pd.to_numeric(dfw[args.feature2_col], errors="coerce")
        dfw = dfw.dropna(subset=[args.feature1_col, args.feature2_col])
    else:
        dfw = dfw.dropna(subset=[args.feature1_col])

    # 同一站点同一时刻若有重复，取均值
    dfw = dfw.groupby(["station_id", "date"], as_index=False).mean(numeric_only=True)

    # 只保留 power 里存在的 station
    dfw = dfw[dfw["station_id"].isin(station_ids)].copy()


    print(f"[FEATURE1] {args.feature1_col}")
    if use_feature2:
        print(f"[FEATURE2] {args.feature2_col}")
        print(f"[MODE] dual-feature")
    else:
        print(f"[MODE] single-feature")
    print(f"[DATA] rows(after groupby/filter)={len(dfw)}")
    print(f"[DATA] station count={dfw['station_id'].nunique()}")


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

    feature_pt = torch.zeros((N, T, D), dtype=out_dtype)

    missing_text = args.missing_text.strip()
    if not missing_text:
        if use_feature2:
            missing_text = (
                f"{args.feature1_name.capitalize()} and {args.feature2_name} data are unavailable "
                f"for this station and time window."
            )
        else:
            missing_text = (
                f"{args.feature1_name.capitalize()} data are unavailable "
                f"for this station and time window."
            )

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
    print(f"[ALLOC] feature_pt shape={tuple(feature_pt.shape)}")

    # ------------------------------------------------------------
    # 4) For each station:
    #    clean hourly -> 15min interpolate -> align to global dt
    #    -> joint token-window text -> embedding
    # ------------------------------------------------------------
    print("[4/5] Generating joint feature embeddings...")

    dt_index = pd.DatetimeIndex(dt)
    global_dates = pd.Series(dt).dt.normalize()

    for sid in station_ids:
        sid_idx = sid2idx[sid]

        g = dfw[dfw["station_id"] == sid].copy()
        if len(g) == 0:
            print(f"[WARN] sid={sid} has no matched rows, fill with missing embedding.")
            feature_pt[sid_idx, :, :] = missing_emb.unsqueeze(0).repeat(T, 1)
            continue

        g = g.sort_values("date")
        g = g.set_index("date")

        s1 = g[args.feature1_col]
        s1_hourly = clean_hourly_series(s1, args.feature1_min, args.feature1_max)
        s1_15 = hourly_to_15min_series(s1_hourly)
        s1_aligned = s1_15.reindex(dt_index).interpolate(method="time").ffill().bfill()

        if use_feature2:
            s2 = g[args.feature2_col]
            s2_hourly = clean_hourly_series(s2, args.feature2_min, args.feature2_max)
            s2_15 = hourly_to_15min_series(s2_hourly)
            s2_aligned = s2_15.reindex(dt_index).interpolate(method="time").ffill().bfill()

        if args.days_limit > 0:
            keep_dates = global_dates.drop_duplicates().iloc[:args.days_limit]
            keep_mask = global_dates.isin(set(keep_dates))
            valid_tidx = np.where(keep_mask.values)[0]
        else:
            valid_tidx = np.arange(T)

        candidate_tidx = [t for t in valid_tidx if t + args.token_len <= T]

        if len(candidate_tidx) == 0:
            print(f"[WARN] sid={sid} has no valid token windows, fill with missing embedding.")
            feature_pt[sid_idx, :, :] = missing_emb.unsqueeze(0).repeat(T, 1)
            continue

        texts = []
        for t in candidate_tidx:
            values1_window = s1_aligned.iloc[t:t + args.token_len].to_numpy(dtype=np.float32)
            start_time = dt.iloc[t]
            end_time = dt.iloc[t + args.token_len - 1]

            if use_feature2:
                values2_window = s2_aligned.iloc[t:t + args.token_len].to_numpy(dtype=np.float32)
                text = build_joint_window_text(
                    values1_window=values1_window,
                    values2_window=values2_window,
                    start_time=start_time,
                    end_time=end_time,
                    feature1_name=args.feature1_name,
                    feature1_unit=args.feature1_unit,
                    feature2_name=args.feature2_name,
                    feature2_unit=args.feature2_unit,
                    feature1_trend_threshold=args.feature1_trend_threshold,
                    feature2_trend_threshold=args.feature2_trend_threshold,
                    text_style=args.text_style,
                    prompt_mode=args.prompt_mode,
                )
            else:
                text = build_single_window_text(
                    values_window=values1_window,
                    start_time=start_time,
                    end_time=end_time,
                    feature_name=args.feature1_name,
                    feature_unit=args.feature1_unit,
                    trend_threshold=args.feature1_trend_threshold,
                    text_style=args.text_style,
                    prompt_mode=args.prompt_mode,
                )

            texts.append(text)

        embs = embed_texts(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        embs = embs.half() if out_dtype == torch.float16 else embs.float()

        feature_pt[sid_idx, :, :] = missing_emb.unsqueeze(0).repeat(T, 1)
        for k, t in enumerate(candidate_tidx):
            feature_pt[sid_idx, t, :] = embs[k]

        if use_feature2:
            print(
                f"[STATION] sid={sid}, valid_windows={len(candidate_tidx)}, "
                f"{args.feature1_name}_mean={float(s1_aligned.mean()):.3f}, "
                f"{args.feature2_name}_mean={float(s2_aligned.mean()):.3f}"
            )
        else:
            print(
                f"[STATION] sid={sid}, valid_windows={len(candidate_tidx)}, "
                f"{args.feature1_name}_mean={float(s1_aligned.mean()):.3f}, "
                f"{args.feature1_name}_min={float(s1_aligned.min()):.3f}, "
                f"{args.feature1_name}_max={float(s1_aligned.max()):.3f}"
            )

    # ------------------------------------------------------------
    # 5) Save
    # ------------------------------------------------------------
    print("[5/5] Saving...")
    out_dir = os.path.dirname(args.out_feature_pt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(feature_pt, args.out_feature_pt)

    nz_ratio = float((feature_pt.abs().sum(dim=-1) != 0).float().mean().item())
    print(f"[OK] saved joint feature.pt -> {args.out_feature_pt}")
    print(f"[OK] shape={tuple(feature_pt.shape)}, dtype={feature_pt.dtype}")
    print(f"[STAT] nonzero ratio={nz_ratio:.6f}")

    nz = (feature_pt.abs().sum(dim=-1) != 0).nonzero(as_tuple=False)
    if nz.numel() > 0:
        i, t = nz[0].tolist()
        print(f"[SAMPLE] first nonzero at sid_idx={i}, tidx={t}, vec_norm={feature_pt[i, t].float().norm().item():.6f}")


if __name__ == "__main__":
    main()
    '''
    1）单变量 + 语言版
温度语义模板
python /root/autotimes/embedding构建/weather/GPT2_weather_time_04013_humid.py \
  --power_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --weather_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/weather_hourly_20210501_20240430_humid.csv \
  --llm_ckp_dir /root/autodl-tmp/hf_models/gpt2 \
  --out_feature_pt /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/language_temperature_0413_gpt2.pt \
  --feature1_col temperature_2m_mean \
  --feature1_name "temperature" \
  --feature1_unit "degrees Celsius" \
  --feature1_min -40 \
  --feature1_max 100 \
  --feature1_trend_threshold 0.5 \
  --prompt_mode semantic \
  --text_style simple \
  --token_len 96 \
  --batch_size 64 \
  --max_length 192 \
  --device cuda \
  --dtype float16
湿度语义模板
python /root/autotimes/embedding构建/weather/GPT2_weather_time_04013_humid.py \
  --power_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --weather_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/weather_hourly_20210501_20240430_humid.csv \
  --llm_ckp_dir /root/autodl-tmp/hf_models/gpt2 \
  --out_feature_pt /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/humidity_single_semantic_token96_gpt2.pt \
  --feature1_col relative_humidity_2m_mean \
  --feature1_name "relative humidity" \
  --feature1_unit "percent" \
  --feature1_min -1 \
  --feature1_max 101 \
  --feature1_trend_threshold 2.0 \
  --prompt_mode semantic \
  --text_style simple \
  --token_len 96 \
  --batch_size 64 \
  --max_length 192 \
  --device cuda \
  --dtype float16
风速语义模板
python /root/autotimes/embedding构建/weather/GPT2_weather_time_04013_humid.py \
  --power_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --weather_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/weather_hourly_20210501_20240430_humid.csv \
  --llm_ckp_dir /root/autodl-tmp/hf_models/gpt2 \
  --out_feature_pt /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/wind_single_semantic_token96_gpt2.pt \
  --feature1_col wind_speed_10m_mean \
  --feature1_name "wind speed" \
  --feature1_unit "km/h" \
  --feature1_min 0 \
  --feature1_max 200 \
  --feature1_trend_threshold 1.0 \
  --prompt_mode semantic \
  --text_style simple \
  --token_len 96 \
  --batch_size 64 \
  --max_length 192 \
  --device cuda \
  --dtype float16
2）单变量 + 数值版
温度数值模板
python /root/autotimes/embedding构建/weather/GPT2_weather_time_04013_humid.py \
  --power_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --weather_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/weather_hourly_20210501_20240430_humid.csv \
  --llm_ckp_dir /root/autodl-tmp/hf_models/gpt2 \
  --out_feature_pt /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/temp_single_numeric_token96_gpt2.pt \
  --feature1_col temperature_2m_mean \
  --feature1_name "temperature" \
  --feature1_unit "degrees Celsius" \
  --feature1_min -40 \
  --feature1_max 100 \
  --feature1_trend_threshold 0.5 \
  --prompt_mode numeric \
  --text_style simple \
  --token_len 96 \
  --batch_size 64 \
  --max_length 192 \
  --device cuda \
  --dtype float16
湿度数值模板
python /root/autotimes/embedding构建/weather/GPT2_weather_time_04013_humid.py \
  --power_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --weather_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/weather_hourly_20210501_20240430_humid.csv \
  --llm_ckp_dir /root/autodl-tmp/hf_models/gpt2 \
  --out_feature_pt /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/humidity_single_numeric_token96_gpt2.pt \
  --feature1_col relative_humidity_2m_mean \
  --feature1_name "relative humidity" \
  --feature1_unit "percent" \
  --feature1_min -1 \
  --feature1_max 101 \
  --feature1_trend_threshold 2.0 \
  --prompt_mode numeric \
  --text_style simple \
  --token_len 96 \
  --batch_size 64 \
  --max_length 192 \
  --device cuda \
  --dtype float16
3）双变量 + 语言版
温度 + 湿度 语义模板
python /root/autotimes/embedding构建/weather/GPT2_weather_time_04013_humid.py \
  --power_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --weather_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/weather_hourly_20210501_20240430_humid.csv \
  --llm_ckp_dir /root/autodl-tmp/hf_models/gpt2 \
  --out_feature_pt /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/temp_humidity_semantic_token96_gpt2.pt \
  --feature1_col temperature_2m_mean \
  --feature1_name "temperature" \
  --feature1_unit "degrees Celsius" \
  --feature1_min -40 \
  --feature1_max 100 \
  --feature1_trend_threshold 0.5 \
  --feature2_col relative_humidity_2m_mean \
  --feature2_name "relative humidity" \
  --feature2_unit "percent" \
  --feature2_min -1 \
  --feature2_max 101 \
  --feature2_trend_threshold 2.0 \
  --prompt_mode semantic \
  --text_style simple \
  --token_len 96 \
  --batch_size 64 \
  --max_length 192 \
  --device cuda \
  --dtype float16
温度 + 风速 语义模板
python /root/autotimes/embedding构建/weather/GPT2_weather_time_04013_humid.py \
  --power_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --weather_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/weather_hourly_20210501_20240430_humid.csv \
  --llm_ckp_dir /root/autodl-tmp/hf_models/gpt2 \
  --out_feature_pt /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/temp_wind_semantic_token96_gpt2.pt \
  --feature1_col temperature_2m_mean \
  --feature1_name "temperature" \
  --feature1_unit "degrees Celsius" \
  --feature1_min -40 \
  --feature1_max 100 \
  --feature1_trend_threshold 0.5 \
  --feature2_col wind_speed_10m_mean \
  --feature2_name "wind speed" \
  --feature2_unit "km/h" \
  --feature2_min 0 \
  --feature2_max 200 \
  --feature2_trend_threshold 1.0 \
  --prompt_mode semantic \
  --text_style simple \
  --token_len 96 \
  --batch_size 64 \
  --max_length 192 \
  --device cuda \
  --dtype float16
4）双变量 + 数值版
温度 + 湿度 数值模板
python /root/autotimes/embedding构建/weather/GPT2_weather_time_04013_humid.py \
  --power_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --weather_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/weather_hourly_20210501_20240430_humid.csv \
  --llm_ckp_dir /root/autodl-tmp/hf_models/gpt2 \
  --out_feature_pt /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/temp_humidity_numeric_token96_gpt2.pt \
  --feature1_col temperature_2m_mean \
  --feature1_name "temperature" \
  --feature1_unit "degrees Celsius" \
  --feature1_min -40 \
  --feature1_max 100 \
  --feature1_trend_threshold 0.5 \
  --feature2_col relative_humidity_2m_mean \
  --feature2_name "relative humidity" \
  --feature2_unit "percent" \
  --feature2_min -1 \
  --feature2_max 101 \
  --feature2_trend_threshold 2.0 \
  --prompt_mode numeric \
  --text_style simple \
  --token_len 96 \
  --batch_size 64 \
  --max_length 192 \
  --device cuda \
  --dtype float16
温度 + 风速 数值模板
python /root/autotimes/embedding构建/weather/GPT2_weather_time_04013_humid.py \
  --power_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --weather_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/weather_hourly_20210501_20240430_humid.csv \
  --llm_ckp_dir /root/autodl-tmp/hf_models/gpt2 \
  --out_feature_pt /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/temp_wind_numeric_token96_gpt2.pt \
  --feature1_col temperature_2m_mean \
  --feature1_name "temperature" \
  --feature1_unit "degrees Celsius" \
  --feature1_min -40 \
  --feature1_max 100 \
  --feature1_trend_threshold 0.5 \
  --feature2_col wind_speed_10m_mean \
  --feature2_name "wind speed" \
  --feature2_unit "km/h" \
  --feature2_min 0 \
  --feature2_max 200 \
  --feature2_trend_threshold 1.0 \
  --prompt_mode numeric \
  --text_style simple \
  --token_len 96 \
  --batch_size 64 \
  --max_length 192 \
  --device cuda \
  --dtype float16
    '''