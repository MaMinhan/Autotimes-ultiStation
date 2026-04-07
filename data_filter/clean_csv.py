import pandas as pd

src = "/root/autodl-tmp/datasets/FY24_long/Aberdeen 66_11kV FY2024_long.csv"
clean_path = "/root/autodl-tmp/datasets/temp.csv"
dropped_path = "/root/autodl-tmp/datasets/temp2.csv"

# 1. 读取原始数据
df = pd.read_csv(src)

# 2. 保留原始副本（用于对比）
df_raw = df.copy()

# 3. 标准化字段
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df["station"] = df["station"].astype(str)

# 4. 标记哪些行是非法的
mask_valid = (
    df["datetime"].notna()
    & df["station"].notna()
    & df["target"].notna()
)

# 5. 拆分
df_clean = df[mask_valid].copy()
df_dropped = df[~mask_valid].copy()

# 6. 排序（保证后续可复现）
df_clean = df_clean.sort_values(
    ["station", "datetime"]
).reset_index(drop=True)

# 7. 保存
df_clean.to_csv(clean_path, index=False)
df_dropped.to_csv(dropped_path, index=False)

# 8. 打印统计信息
print("===== CLEAN SUMMARY =====")
print("Original rows :", len(df_raw))
print("Clean rows    :", len(df_clean))
print("Dropped rows  :", len(df_dropped))
print()
print("Saved clean   :", clean_path)
print("Saved dropped :", dropped_path)
