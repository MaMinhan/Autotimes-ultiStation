import pandas as pd
import geopandas as gpd

# =======================
# 你需要改的 3 个路径
# =======================
STATIONS_CSV = r"stations_to_SA2_SA3_SA4_2021.csv"   # 你的站点映射表（含 SA2_CODE21）
CENSUS_SA2_CSV = r"C:\Users\maminhan\Desktop\AusgridSA2\2021 Census GCP Statistical Area 2 for NSW\2021Census_G01_NSW_SA2.csv"             # Census GCP SA2（含 SA2_CODE_2021 和 Tot_P_P 等）
SA2_SHP = r"C:\Users\maminhan\Desktop\AusgridSA2\SA1234元数据shp2021\SA2_2021_AUST_GDA2020.shp"               # ASGS 2021 SA2 边界（含 geometry）
OUT_CSV = r"stations_exogenous_features.csv"

# =======================
# 1) 读取数据
# =======================
stations = pd.read_csv(STATIONS_CSV)
census = pd.read_csv(CENSUS_SA2_CSV)

# 统一 SA2 code 类型：全部用字符串，避免 merge 失败
stations["SA2_CODE21"] = stations["SA2_CODE21"].astype(str).str.strip()
census["SA2_CODE_2021"] = census["SA2_CODE_2021"].astype(str).str.strip()

# =======================
# 2) 取 SA2 面积（km²）
#    优先用 shp 自带 AREASQKM21（若存在），否则用 geometry 计算
# =======================
sa2 = gpd.read_file(SA2_SHP)

area_col = None
for c in sa2.columns:
    if c.upper() in ["AREASQKM21", "AREASQKM_21", "AREASQKM"]:
        area_col = c
        break

if area_col:
    sa2_area = sa2[["SA2_CODE21", area_col]].copy()
    sa2_area.rename(columns={area_col: "area_sqkm"}, inplace=True)
else:
    # 用澳洲常用等积投影算面积（避免经纬度直接算面积）
    sa2_proj = sa2.to_crs(epsg=3577)  # Australian Albers
    sa2_proj["area_sqkm"] = sa2_proj.geometry.area / 1e6
    sa2_area = sa2_proj[["SA2_CODE21", "area_sqkm"]].copy()

sa2_area["SA2_CODE21"] = sa2_area["SA2_CODE21"].astype(str).str.strip()

# =======================
# 3) Census：提取你要的字段（人口、年龄段、居住类型）
# =======================
needed_cols = [
    "SA2_CODE_2021",
    "Tot_P_P",
    "Age_0_4_yr_P", "Age_5_14_yr_P",
    "Age_15_19_yr_P", "Age_20_24_yr_P",
    "Age_25_34_yr_P", "Age_35_44_yr_P",
    "Age_45_54_yr_P", "Age_55_64_yr_P",
    "Age_65_74_yr_P", "Age_75_84_yr_P",
    "Age_85ov_P",
    "Count_psns_occ_priv_dwgs_P",
    "Count_Persons_other_dwgs_P",
]

missing = [c for c in needed_cols if c not in census.columns]
if missing:
    raise ValueError(
        "Census 文件缺少以下列（可能你下载的不是同一套 GCP 或列名略不同）：\n"
        + "\n".join(missing)
    )

census_sel = census[needed_cols].drop_duplicates(subset=["SA2_CODE_2021"]).copy()

# 强制数值列为数值（防止读成字符串）
for c in needed_cols:
    if c != "SA2_CODE_2021":
        census_sel[c] = pd.to_numeric(census_sel[c], errors="coerce")

# =======================
# 4) 合并：站点表 + Census + 面积
# =======================
df = stations.merge(
    census_sel, left_on="SA2_CODE21", right_on="SA2_CODE_2021", how="left"
).merge(
    sa2_area, on="SA2_CODE21", how="left"
)

# =======================
# 5) 计算你要的外生变量
# =======================
# 人口密度
df["pop_density_per_sqkm"] = df["Tot_P_P"] / df["area_sqkm"]

# 年龄结构（比例）
df["age_0_14"] = df["Age_0_4_yr_P"] + df["Age_5_14_yr_P"]
df["age_15_64"] = (
    df["Age_15_19_yr_P"] + df["Age_20_24_yr_P"] + df["Age_25_34_yr_P"] +
    df["Age_35_44_yr_P"] + df["Age_45_54_yr_P"] + df["Age_55_64_yr_P"]
)
df["age_65_plus"] = df["Age_65_74_yr_P"] + df["Age_75_84_yr_P"] + df["Age_85ov_P"]

df["young_ratio_0_14"] = df["age_0_14"] / df["Tot_P_P"]
df["working_ratio_15_64"] = df["age_15_64"] / df["Tot_P_P"]
df["elderly_ratio_65_plus"] = df["age_65_plus"] / df["Tot_P_P"]

# 居住类型比例（比例）
df["private_dwelling_ratio"] = df["Count_psns_occ_priv_dwgs_P"] / df["Tot_P_P"]
df["other_dwelling_ratio"] = df["Count_Persons_other_dwgs_P"] / df["Tot_P_P"]

# =======================
# 6) 只保留输出列（按你的需求精简）
# =======================
out_cols = [
    "station_clean", "lat", "lon",
    "SA2_CODE21", "SA2_NAME21",
    "SA3_CODE21", "SA3_NAME21",
    "SA4_CODE21", "SA4_NAME21",
    "Tot_P_P", "area_sqkm", "pop_density_per_sqkm",
    "young_ratio_0_14", "working_ratio_15_64", "elderly_ratio_65_plus",
    "private_dwelling_ratio", "other_dwelling_ratio",
]

# 有些站点表可能没有 SA3/SA4 列名（你可以按实际删掉）
out_cols = [c for c in out_cols if c in df.columns]

out = df[out_cols].copy()
out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("✅ 输出完成：", OUT_CSV)
print("未匹配到人口(Tot_P_P)的行数:", out["Tot_P_P"].isna().sum() if "Tot_P_P" in out.columns else "N/A")
print("未匹配到面积(area_sqkm)的行数:", out["area_sqkm"].isna().sum() if "area_sqkm" in out.columns else "N/A")
print(out.head(10))
