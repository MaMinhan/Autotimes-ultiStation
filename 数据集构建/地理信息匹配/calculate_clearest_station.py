import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# 1) 读 SA2（也可顺便带上 SA3/SA4 字段）
sa2 = gpd.read_file("SA1234元数据shp2021/SA2_2021_AUST_GDA2020.shp")[[
    "SA2_CODE21","SA2_NAME21",
    "SA3_CODE21","SA3_NAME21",
    "SA4_CODE21","SA4_NAME21",
    "geometry"
]]

print("SA2 CRS:", sa2.crs)

# 2) 读站点坐标
df = pd.read_csv("stations_with_latlon.csv")
df = df.dropna(subset=["lat","lon"]).copy()

pts = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])],
    crs="EPSG:4326"
).to_crs(sa2.crs)

# 3) 点落面（用 intersects 更稳）
joined = gpd.sjoin(pts, sa2, predicate="intersects", how="left")

# 4) 如果有少量没匹配到：用最近多边形兜底
miss = joined["SA2_CODE21"].isna()
if miss.any():
    # 建空间索引加速最近邻
    sa2_sindex = sa2.sindex

    def nearest_sa2(row):
        geom = row.geometry
        # 先用 bbox 候选，减少计算量
        cand_idx = list(sa2_sindex.nearest(geom, 1))
        cand = sa2.iloc[cand_idx[0]]
        return pd.Series([cand["SA2_CODE21"], cand["SA2_NAME21"],
                          cand["SA3_CODE21"], cand["SA3_NAME21"],
                          cand["SA4_CODE21"], cand["SA4_NAME21"]])

    joined.loc[miss, ["SA2_CODE21","SA2_NAME21","SA3_CODE21","SA3_NAME21","SA4_CODE21","SA4_NAME21"]] = \
        joined[miss].apply(nearest_sa2, axis=1)

# 5) 输出最终映射表
out = joined.drop(columns=["geometry", "index_right"], errors="ignore")
out.to_csv("stations_to_SA2_SA3_SA4_2021.csv", index=False, encoding="utf-8-sig")
print("✅ 已生成 stations_to_SA2_SA3_SA4_2021.csv")
