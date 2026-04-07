import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# -------------------------
# 1. 读取站点坐标
# -------------------------
nodes = pd.read_csv("substation_coordinates.csv")

g_nodes = gpd.GeoDataFrame(
    nodes,
    geometry=[Point(xy) for xy in zip(nodes.lon, nodes.lat)],
    crs="EPSG:4326"
)

# -------------------------
# 2. 读取 SA2（必须指定 layer）
# -------------------------
sa2 = gpd.read_file(
    "ASGS 2016 Volume 1.gpkg",
    layer="SA2_2016_AUST"
)

# ⚠️ 统一 CRS
sa2 = sa2.to_crs("EPSG:4326")

# -------------------------
# 3. Spatial Join（点 → SA2）
# -------------------------
joined = gpd.sjoin(
    g_nodes,
    sa2,
    how="left",
    predicate="within"
)

# -------------------------
# 4. 计算 SA2 面积（km²）
# -------------------------
sa2_area = sa2.to_crs("EPSG:3577")  # Australian Albers
sa2_area["area_km2"] = sa2_area.geometry.area / 1e6

joined = joined.merge(
    sa2_area[["SA2_MAINCODE_2016", "area_km2"]],
    on="SA2_MAINCODE_2016",
    how="left"
)

# -------------------------
# 5. 合并人口数据
# -------------------------
census = pd.read_csv("2016 Census GCP Statistical Area 2 for NSW\2016Census_G01_NSW_SA2.csv")
joined = joined.merge(
    census,
    on="SA2_MAINCODE_2016",
    how="left"
)

# -------------------------
# 6. 计算人口密度
# -------------------------
joined["pop_density"] = joined["population"] / joined["area_km2"]

# -------------------------
# 7. 输出结果
# -------------------------
joined[[
    "station",
    "lat",
    "lon",
    "SA2_MAINCODE_2016",
    "SA2_NAME_2016",
    "population",
    "area_km2",
    "pop_density"
]].to_csv("station_population_density.csv", index=False)

print("✅ Done! Output saved as station_population_density.csv")
