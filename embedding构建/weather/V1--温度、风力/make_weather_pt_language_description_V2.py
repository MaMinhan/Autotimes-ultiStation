import os
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden_state: [B, L, D]
    attention_mask:    [B, L]
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B, L, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                   # [B, D]
    denom = mask.sum(dim=1).clamp(min=1e-6)                          # [B, 1]
    return summed / denom


def make_bins(series: pd.Series, qs):
    """Return quantile thresholds used for bucketing numeric weather into language-only labels."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return [0.0] * len(qs)
    return [float(s.quantile(q)) for q in qs]


def bucket(x, bins, labels):
    """
    Map numeric x -> one discrete label.
    labels length must be len(bins)+1.
    """
    if pd.isna(x):
        return "unknown"
    x = float(x)
    for i, th in enumerate(bins):
        if x <= th:
            return labels[i]
    return labels[-1]


def build_weather_text_language(row: pd.Series, cfg: dict) -> str:
    """
    Build a *language-only* weather prompt.

    Requirements:
      - MUST include station_id and date (date kept as 'YYYY-MM-DD')
      - MUST NOT include any weather numeric values (only discrete labels)
    """
    sid = int(row["station_id"])
    d = pd.to_datetime(row["date"]).date()  # str(d) -> YYYY-MM-DD

    # Read numeric columns (never inserted into text)
    tmean = row.get("temperature_2m_mean", np.nan)
    rain  = row.get("precipitation_sum", np.nan)
    wind  = row.get("wind_speed_10m_mean", np.nan)
    rad   = row.get("shortwave_radiation_sum", np.nan)

    # Discretize into labels
    t_lv = bucket(tmean, cfg["t_bins"], cfg["t_labels"])
    r_lv = bucket(rain,  cfg["r_bins"], cfg["r_labels"])
    w_lv = bucket(wind,  cfg["w_bins"], cfg["w_labels"])
    s_lv = bucket(rad,   cfg["s_bins"], cfg["s_labels"])

    # Optional coarse summary using a controlled vocabulary
    overall_parts = []
    if t_lv in ["hot", "warm"]:
        overall_parts.append("warm")
    elif t_lv in ["very cold", "cold"]:
        overall_parts.append("cold")
    else:
        overall_parts.append("mild")

    if r_lv in ["heavy rain", "moderate rain"]:
        overall_parts.append("rainy")
    elif r_lv == "light rain":
        overall_parts.append("slightly rainy")
    else:
        overall_parts.append("dry")

    if w_lv in ["windy", "strong wind"]:
        overall_parts.append("windy")

    if s_lv in ["high", "very high"]:
        overall_parts.append("sunny")
    elif s_lv == "low":
        overall_parts.append("cloudy")

    overall = " and ".join(dict.fromkeys(overall_parts))  # de-duplicate, keep order

    # IMPORTANT: no numeric weather values appear below.
    text = (
        f"Weather report. Station {sid}. Date {d}. "
        f"Mean temperature is {t_lv}. "
        f"Precipitation is {r_lv}. "
        f"Wind is {w_lv}. "
        f"Solar radiation is {s_lv}. "
        f"Overall: {overall}. "
        f"This weather may influence electricity demand."
    )
    return text


@torch.no_grad()
def embed_texts(texts, tokenizer, model, device, batch_size=64, max_length=128):
    """Embed list[str] -> tensor [M, D] using mean pooling."""
    embs = []
    model.eval()
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
        e = mean_pool(out.last_hidden_state, enc["attention_mask"])  # [B, D]
        embs.append(e.detach().cpu())
    return torch.cat(embs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power_csv", type=str, required=True,
                    help="Electricity long-format CSV (must include: datetime, station_id). Used to build the global time axis.")
    ap.add_argument("--weather_csv", type=str, required=True,
                    help="Weather daily CSV (must include: date, station_id, weather columns).")
    ap.add_argument("--llm_ckp_dir", type=str, required=True,
                    help="HF model directory (e.g., gpt2 or your local checkpoint).")
    ap.add_argument("--time_pt_path", type=str, default="",
                    help="Optional: time_only.pt path to validate T alignment.")
    ap.add_argument("--out_pt", type=str, required=True, help="Output weather.pt path")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    args = ap.parse_args()

    # --- 1) Read power CSV to build global time axis ---
    dfp = pd.read_csv(args.power_csv, usecols=["datetime", "station_id"])
    dfp["datetime"] = pd.to_datetime(dfp["datetime"], errors="coerce")
    dfp = dfp.dropna(subset=["datetime", "station_id"])
    dfp["station_id"] = dfp["station_id"].astype(int)

    dt = dfp["datetime"].drop_duplicates().sort_values().reset_index(drop=True)
    T = len(dt)

    dt_date = pd.to_datetime(dt).dt.normalize()  # [T], normalized date per slot
    tmp = pd.DataFrame({"tidx": np.arange(T), "date": dt_date})
    date2tidx = {d: g["tidx"].to_numpy() for d, g in tmp.groupby("date")}

    station_ids = sorted(dfp["station_id"].unique().tolist())
    N = len(station_ids)
    sid2idx = {sid: i for i, sid in enumerate(station_ids)}

    # optional: validate time_pt length
    if args.time_pt_path:
        time_pt = torch.load(args.time_pt_path, map_location="cpu")
        if time_pt.shape[0] != T:
            raise ValueError(f"time_pt T mismatch: time_pt={time_pt.shape[0]} vs power_dt={T}")

    # --- 2) Read weather CSV (daily) ---
    # Hard-code weather columns used by the language template.
    weather_cols = [
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_mean",
        "shortwave_radiation_sum",
    ]
    usecols = ["date", "station_id"] + weather_cols
    dfw = pd.read_csv(args.weather_csv, usecols=usecols)
    dfw["date"] = pd.to_datetime(dfw["date"], errors="coerce")
    dfw = dfw.dropna(subset=["date", "station_id"])
    dfw["station_id"] = dfw["station_id"].astype(int)

    # Aggregate duplicates per (sid, date)
    dfw = dfw.groupby(["station_id", "date"], as_index=False).mean(numeric_only=True)

    # --- 3) Build quantile bins + label vocab ---
    cfg = {
        "t_labels": ["very cold", "cold", "mild", "warm", "hot"],              # 4 bins -> 5 labels
        "r_labels": ["dry", "light rain", "moderate rain", "heavy rain"],     # 3 bins -> 4 labels
        "w_labels": ["calm", "breeze", "windy", "strong wind"],               # 3 bins -> 4 labels
        "s_labels": ["low", "medium", "high", "very high"],                   # 3 bins -> 4 labels
        "t_bins": make_bins(dfw["temperature_2m_mean"], [0.2, 0.4, 0.6, 0.8]),
        "r_bins": make_bins(dfw["precipitation_sum"],   [0.5, 0.8, 0.95]),
        "w_bins": make_bins(dfw["wind_speed_10m_mean"], [0.5, 0.8, 0.95]),
        "s_bins": make_bins(dfw["shortwave_radiation_sum"], [0.25, 0.5, 0.75]),
    }

    # --- 4) Load tokenizer/model and allocate output ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_ckp_dir)
    model = AutoModel.from_pretrained(args.llm_ckp_dir).to(device)

    # GPT2 often has no pad_token; set to eos_token for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    D = model.config.hidden_size
    out_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    weather_pt = torch.zeros((N, T, D), dtype=out_dtype)

    filled = 0
    total_slots = N * T

    # --- 5) For each (sid, date), embed prompt and broadcast to all tidx of that date ---
    for sid, g in dfw.groupby("station_id"):
        if sid not in sid2idx:
            continue
        sid_idx = sid2idx[sid]

        # keep only dates that exist in the power time axis
        g = g[g["date"].isin(date2tidx.keys())].copy()
        if len(g) == 0:
            continue

        texts = [build_weather_text_language(row, cfg) for _, row in g.iterrows()]
        embs = embed_texts(
            texts, tokenizer, model, device,
            batch_size=args.batch_size, max_length=args.max_length
        )  # [M, D]

        embs = embs.half() if out_dtype == torch.float16 else embs.float()

        for k, (_, row) in enumerate(g.iterrows()):
            d = row["date"]
            tidx = date2tidx[d]
            weather_pt[sid_idx, tidx, :] = embs[k]  # broadcast
            filled += len(tidx)

    # --- 6) Save + coverage stats ---
    out_dir = os.path.dirname(args.out_pt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(weather_pt, args.out_pt)

    nonzero_ratio = float((weather_pt.abs().sum(dim=-1) != 0).float().mean().item())
    print(f"[OK] saved weather.pt: shape={tuple(weather_pt.shape)} dtype={weather_pt.dtype} -> {args.out_pt}")
    print(f"[STAT] nonzero_ratio={nonzero_ratio:.6f} filled_slots={filled} / total_slots={total_slots}")

    # Optional quick sanity sample
    if N > 0 and T > 0:
        nz = (weather_pt.abs().sum(dim=-1) != 0).nonzero(as_tuple=False)
        if nz.numel() > 0:
            i, t = nz[0].tolist()
            print(f"[SAMPLE] first nonzero at (sid_idx={i}, tidx={t}), vec_norm={weather_pt[i,t].float().norm().item():.6f}")


if __name__ == "__main__":
    main()