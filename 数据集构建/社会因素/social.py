import re
import pandas as pd

# =========================
# 输入路径
# =========================
station_sa_csv = "/root/autodl-tmp/datasets/SelfMadeAusgridData/stations_to_SA2_SA3_SA4_2021.csv"
station_map_csv = "/root/autodl-tmp/datasets/SelfMadeAusgridData/station_map.csv"
social_csv = "/root/autodl-tmp/datasets/SelfMadeAusgridData/social/stations_exogenous_features.csv"

# =========================
# 输出路径
# =========================
output_csv = "/root/autodl-tmp/datasets/SelfMadeAusgridData/social/station_social_normalized.csv"

# =========================
# 1. 读取三个文件
# =========================
df_station_sa = pd.read_csv(station_sa_csv)
df_station_map = pd.read_csv(station_map_csv)
df_social = pd.read_csv(social_csv)

# 去掉列名空格
df_station_sa.columns = df_station_sa.columns.str.strip()
df_station_map.columns = df_station_map.columns.str.strip()
df_social.columns = df_social.columns.str.strip()

# 去掉字符串字段首尾空格
for col in df_station_sa.columns:
    if df_station_sa[col].dtype == "object":
        df_station_sa[col] = df_station_sa[col].astype(str).str.strip()

for col in df_station_map.columns:
    if df_station_map[col].dtype == "object":
        df_station_map[col] = df_station_map[col].astype(str).str.strip()

for col in df_social.columns:
    if df_social[col].dtype == "object":
        df_social[col] = df_social[col].astype(str).str.strip()

# =========================
# 2. station_map:
#    保留完整站点名，同时提取基础站点名用于匹配 station_clean
# =========================
# 假设 station_map.csv 格式:
# station,station_id
# Aberdeen 66_11kV,0
# Adamstown 132_11kV,1

df_station_map = df_station_map.rename(columns={
    "station": "stationname",
    "station_id": "stationid"
})

# 提取基础站点名：
# "Aberdeen 66_11kV" -> "Aberdeen"
# "Adamstown 132_11kV" -> "Adamstown"
def extract_station_clean(name: str) -> str:
    name = str(name).strip()
    # 去掉末尾类似 " 66_11kV" / " 132_11kV" / " 33_11kV"
    name = re.sub(r"\s+\d+_\d+kV$", "", name, flags=re.IGNORECASE)
    return name.strip()

df_station_map["station_clean"] = df_station_map["stationname"].apply(extract_station_clean)

# =========================
# 3. station_clean -> SA2_CODE21
# =========================
# 假设 stations_to_SA2_SA3_SA4_2021.csv 有列:
# station_clean, SA2_CODE21, ...
df_station_sa = df_station_sa.rename(columns={
    "SA2_CODE21": "SA2code"
})

# 只保留需要的映射列，并去重
df_station_sa_map = df_station_sa[["station_clean", "SA2code"]].drop_duplicates()

# merge：station_map(完整站点名) + station_clean 对应 SA2code
df = df_station_map.merge(
    df_station_sa_map,
    on="station_clean",
    how="left"
)

# 检查 station_clean -> SA2code 是否匹配成功
missing_sa2 = df["SA2code"].isna().sum()
print(f"[检查] 未匹配到 SA2code 的站点数: {missing_sa2}")
if missing_sa2 > 0:
    print(df.loc[df["SA2code"].isna(), ["stationname", "station_clean"]].drop_duplicates().head(20))

# =========================
# 4. SA2code -> 社会因素
# =========================
# 假设 social 文件里也有 SA2_CODE21
df_social = df_social.rename(columns={
    "SA2_CODE21": "SA2code"
})

# 转数值，避免类型不一致
df["SA2code"] = pd.to_numeric(df["SA2code"], errors="coerce")
df_social["SA2code"] = pd.to_numeric(df_social["SA2code"], errors="coerce")

# 你关注的 6 个指标
feature_cols = [
    "pop_density_per_sqkm",
    "young_ratio_0_14",
    "working_ratio_15_64",
    "elderly_ratio_65_plus",
    "private_dwelling_ratio",
    "other_dwelling_ratio",
]

# 检查 social 文件中这些列是否存在
missing_cols = [c for c in feature_cols if c not in df_social.columns]
if len(missing_cols) > 0:
    raise ValueError(f"social 文件缺少这些列: {missing_cols}")

# 只取需要列，并按 SA2code 去重
df_social_sub = df_social[["SA2code"] + feature_cols].drop_duplicates(subset=["SA2code"])

# 合并社会因素
df = df.merge(
    df_social_sub,
    on="SA2code",
    how="left"
)

# 检查 SA2code -> 社会因素是否匹配成功
missing_social = df[feature_cols].isna().all(axis=1).sum()
print(f"[检查] 未匹配到社会因素的站点数: {missing_social}")
if missing_social > 0:
    print(df.loc[df[feature_cols].isna().all(axis=1), ["stationname", "station_clean", "SA2code"]].drop_duplicates().head(20))

# =========================
# 5. 对 6 个指标做 Min-Max 归一化
# =========================
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    cmin = df[col].min()
    cmax = df[col].max()

    norm_col = col + "_norm"
    if pd.isna(cmin) or pd.isna(cmax):
        df[norm_col] = pd.NA
    elif cmax == cmin:
        df[norm_col] = 0.0
    else:
        df[norm_col] = (df[col] - cmin) / (cmax - cmin)

# =========================
# 6. 组织最终输出
# =========================
result_cols = [
    "stationname",
    "stationid",
    "SA2code",
    "pop_density_per_sqkm_norm",
    "young_ratio_0_14_norm",
    "working_ratio_15_64_norm",
    "elderly_ratio_65_plus_norm",
    "private_dwelling_ratio_norm",
    "other_dwelling_ratio_norm",
]

result_df = df[result_cols].copy()

# stationid 转成整数显示（如果可转）
result_df["stationid"] = pd.to_numeric(result_df["stationid"], errors="coerce").astype("Int64")
result_df["SA2code"] = pd.to_numeric(result_df["SA2code"], errors="coerce").astype("Int64")

# 去重
result_df = result_df.drop_duplicates(subset=["stationname", "stationid"]).sort_values("stationid")

# =========================
# 7. 保存
# =========================
result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"已保存到: {output_csv}")
print(result_df.head(10))
print("结果表形状:", result_df.shape)