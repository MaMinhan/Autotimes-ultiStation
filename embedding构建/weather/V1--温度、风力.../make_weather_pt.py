import os
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: [B, L, D], attention_mask: [B, L]
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B, L, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                   # [B, D]
    denom = mask.sum(dim=1).clamp(min=1e-6)                          # [B, 1]
    return summed / denom


def build_weather_text(row, weather_cols):
    # 你可以按论文风格自由改模板，这里给一个稳定、信息密度足够的版本
    parts = [f"Station {int(row['station_id'])}. Date {row['date'].date()}."]
    for c in weather_cols:
        v = row[c]
        if pd.isna(v):
            parts.append(f"{c} is missing.")
        else:
            parts.append(f"{c} is {float(v):.3f}.")
    parts.append("These weather conditions may affect electricity demand.")
    return " ".join(parts)


@torch.no_grad()
def embed_texts(texts, tokenizer, model, device, batch_size=64, max_length=128):
    embs = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        # GPT2 没有 pooler_output，常用 mean pooling
        e = mean_pool(out.last_hidden_state, enc["attention_mask"])  # [B, D]
        embs.append(e.detach().cpu())
    return torch.cat(embs, dim=0)  # [M, D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power_csv", type=str, required=True, help="电力 long-format CSV（含 datetime, station_id, target）")
    ap.add_argument("--weather_csv", type=str, required=True, help="天气 CSV（含 date, station_id, weather_cols）")
    ap.add_argument("--llm_ckp_dir", type=str, required=True, help="HF 模型目录（例如 gpt2）")
    ap.add_argument("--time_pt_path", type=str, default="", help="可选：用来校验 T 是否一致（time_only.pt）")
    ap.add_argument("--weather_cols", type=str, required=True,
                    help="逗号分隔天气列名，例如 temperature_2m_mean,precipitation_sum,...")
    ap.add_argument("--out_pt", type=str, required=True, help="输出 weather.pt 路径")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    args = ap.parse_args()

    weather_cols = [c.strip() for c in args.weather_cols.split(",") if c.strip()]
    assert len(weather_cols) > 0

    # 1) 读 power，构造全局 dt 与 station_id 列表
    dfp = pd.read_csv(args.power_csv, usecols=["datetime", "station_id"])
    dfp["datetime"] = pd.to_datetime(dfp["datetime"], errors="coerce")
    dfp = dfp.dropna(subset=["datetime", "station_id"])
    dfp["station_id"] = dfp["station_id"].astype(int)

    dt = dfp["datetime"].drop_duplicates().sort_values().reset_index(drop=True)
    T = len(dt)
    dt_date = pd.to_datetime(dt).dt.normalize()  # [T]
    date2tidx = {}  # date -> np.array(time_indices)
    # 用 pandas 反向索引更快
    tmp = pd.DataFrame({"tidx": np.arange(T), "date": dt_date})
    for d, g in tmp.groupby("date"):
        date2tidx[d] = g["tidx"].to_numpy()

    station_ids = sorted(dfp["station_id"].unique().tolist())
    N = len(station_ids)
    sid2idx = {sid: i for i, sid in enumerate(station_ids)}

    # 可选校验 time_only.pt
    if args.time_pt_path:
        time_pt = torch.load(args.time_pt_path, map_location="cpu")
        if time_pt.shape[0] != T:
            raise ValueError(f"time_pt T mismatch: time_pt={time_pt.shape[0]} vs power_dt={T}")

    # 2) 读 weather（日频）
    usecols = ["date", "station_id"] + weather_cols
    dfw = pd.read_csv(args.weather_csv, usecols=usecols)
    dfw["date"] = pd.to_datetime(dfw["date"], errors="coerce")
    dfw = dfw.dropna(subset=["date", "station_id"])
    dfw["station_id"] = dfw["station_id"].astype(int)

    # 如果同一 (sid, date) 有重复，做均值聚合
    dfw = dfw.groupby(["station_id", "date"], as_index=False).mean(numeric_only=True)

    # 3) 初始化输出张量 [N, T, D]，先不知道 D，等模型出来
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.llm_ckp_dir)
    model = AutoModel.from_pretrained(args.llm_ckp_dir).to(device)

    # GPT2 没 pad_token，需要补
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    D = model.config.hidden_size
    out_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    weather_pt = torch.zeros((N, T, D), dtype=out_dtype)  # 默认 0；你也可以改成 nan 再填

    # 4) 对每个 (sid, date) 编码，然后 broadcast 到该 date 的所有 tidx
    # 为了加速：按 sid 分组后批量 embed 文本
    for sid, g in dfw.groupby("station_id"):
        if sid not in sid2idx:
            continue
        sid_idx = sid2idx[sid]

        # 只保留 power 时间轴中出现过的日期
        g = g[g["date"].isin(date2tidx.keys())].copy()
        if len(g) == 0:
            continue

        texts = [build_weather_text(row, weather_cols) for _, row in g.iterrows()]
        embs = embed_texts(texts, tokenizer, model, device,
                           batch_size=args.batch_size, max_length=args.max_length)  # [M, D]
        if out_dtype == torch.float16:
            embs = embs.half()
        else:
            embs = embs.float()

        for k, (_, row) in enumerate(g.iterrows()):
            d = row["date"]
            tidx = date2tidx[d]
            weather_pt[sid_idx, tidx, :] = embs[k]  # broadcast

    # 5) 保存
    os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
    torch.save(weather_pt, args.out_pt)
    print(f"[OK] saved weather.pt: shape={tuple(weather_pt.shape)} -> {args.out_pt}")


if __name__ == "__main__":
    main()
