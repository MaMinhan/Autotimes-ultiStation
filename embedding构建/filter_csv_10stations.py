import pandas as pd

# ===== 1. 路径 =====
input_csv = "/root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv"
output_csv = "/root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/load_10stations_20230101_20240101.csv"

# ===== 2. 读取 =====
df = pd.read_csv(input_csv)

# 确保列名正确
expected_cols = ["datetime","station_id","target"]
for c in expected_cols:
    if c not in df.columns:
        raise ValueError(f"缺少列: {c}, 当前列名: {df.columns.tolist()}")

# ===== 3. 处理时间 =====
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime"])

# 时间范围: [2023-01-01, 2024-01-01)
start_time = pd.Timestamp("2023-01-01")
end_time = pd.Timestamp("2024-01-01")

df = df[(df["datetime"] >= start_time) & (df["datetime"] < end_time)]

# ===== 4. 选 10 个站点 =====
# 方案A：直接取排序后的前10个 sid
selected_sids = sorted(df["station_id"].dropna().unique())[:10]

# 如果你想手动指定站点，就改成这样：
# selected_sids = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]

df = df[df["station_id"].isin(selected_sids)]

# ===== 5. 只保留需要的列并排序 =====
df = df.sort_values(["station_id", "datetime"]).reset_index(drop=True)
# ===== 6. 保存 =====
df.to_csv(output_csv, index=False)

print("保存完成:", output_csv)
print("shape =", df.shape)
print("站点数 =", df["station_id"].nunique())
print("时间范围 =", df["datetime"].min(), "->", df["datetime"].max())
print("选中的 sid =", selected_sids)