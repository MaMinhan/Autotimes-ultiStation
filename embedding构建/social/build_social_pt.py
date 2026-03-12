import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# ========== 你需要改的 3 个路径 ==========
STATION_SA2_CSV = "/root/autodl-tmp/datasets/SelfMadeAusgridData/stations_to_SA2_SA3_SA4_2021.csv"   # 你这份：station_clean, lat, lon, SA2_CODE21...
SA2_SOCIAL_CSV  = "/root/autodl-tmp/datasets/SelfMadeAusgridData/stations_exogenous_features.csv"    # 你之前那份：SA2_CODE21, Tot_P_P, pop_density...
OUT_PT          = "/root/autodl-tmp/datasets/SelfMadeAusgridData/social_numbers.pt"
OUT_META        = "/root/autodl-tmp/datasets/SelfMadeAusgridData/social_numbers_meta.json"  # 可选：保存列名、station顺序
# ======================================

# 你社会属性里要用的数值列（按你的字段名来）
SOCIAL_COLS = [
    " Tot_P_P",
    " area_sqkm",
    " pop_density_per_sqkm  ",
    " young_ratio_0_14    ",
    " working_ratio_15_64",
    " elderly_ratio_65_plus",
    " private_dwelling_ratio",
    " other_dwelling_ratio",
]

def main():
    # 1) 读 station->SA2 映射表
    st = pd.read_csv(STATION_SA2_CSV)
    st["station_clean    "] = st["station_clean    "].astype(str).str.strip()
    st["SA2_CODE21"] = pd.to_numeric(st["SA2_CODE21"], errors="coerce").astype("Int64")

    # 2) 读 SA2 社会属性表
    sa2 = pd.read_csv(SA2_SOCIAL_CSV)
    sa2["SA2_CODE21"] = pd.to_numeric(sa2["SA2_CODE21"], errors="coerce").astype("Int64")

    # 校验列是否存在
    missing_cols = [c for c in SOCIAL_COLS if c not in sa2.columns]
    if missing_cols:
        raise ValueError(f"SA2 社会表缺少列：{missing_cols}")

    # 3) join：每个 station 拿到它所属 SA2 的社会属性
    merged = st.merge(sa2[["SA2_CODE21"] + SOCIAL_COLS], on="SA2_CODE21", how="left")

    # 4) 处理缺失（很重要：有些 station 可能 SA2 没对应上）
    miss_station = merged[merged[SOCIAL_COLS].isna().any(axis=1)]["station_clean    "].unique().tolist()
    if miss_station:
        print(f"[WARN] 有 {len(miss_station)} 个 station 的社会属性缺失（join 不上 SA2 或 SA2表缺失）：")
        print(miss_station[:20], "..." if len(miss_station) > 20 else "")
    # 用全局中位数填补（比 0 更稳）
    for c in SOCIAL_COLS:
        med = merged[c].median(skipna=True)
        merged[c] = merged[c].fillna(med)

    # 5) 按 AutoTimes 里 station 的排序规则对齐（你 Dataset_MultiStation_Custom 用的是 sorted(unique)）
    station_order = sorted(merged["station_clean    "].unique().tolist())
    merged = merged.set_index("station_clean    ").loc[station_order].reset_index()

    # 6) 标准化（强烈建议，否则尺度差异会很大：人口/面积/比例混一起）
    X = merged[SOCIAL_COLS].astype("float32").to_numpy()
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X).astype("float32")

    # 7) 保存 pt：shape [N_station, d_social]
    social_tensor = torch.tensor(Xn, dtype=torch.float32)
    os.makedirs(os.path.dirname(OUT_PT), exist_ok=True)
    torch.save(social_tensor, OUT_PT)

    # 8) 可选：保存元信息（方便你 debug 对齐是否正确）
    try:
        import json
        meta = {
            "station_order": station_order,
            "social_cols": SOCIAL_COLS,
            "pt_shape": list(social_tensor.shape),
            "note": "Aligned by sorted(station_clean); values are z-scored(StandardScaler)."
        }
        with open(OUT_META, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[WARN] meta 写入失败：", e)

    print("[OK] saved:", OUT_PT, "shape=", tuple(social_tensor.shape))

if __name__ == "__main__":
    main()