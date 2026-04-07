import pandas as pd

# ====== 路径 ======
station_map_path = "/root/autodl-tmp/datasets/SelfMadeAusgridData/station_map.csv"      # 站点映射表
weather_path = "/root/autodl-tmp/datasets/SelfMadeAusgridData/station_weather_daily_2023.csv"               # 原始天气数据
output_path = "/root/autodl-tmp/datasets/SelfMadeAusgridData/weather_with_station_id_2023.csv"

df_map = pd.read_csv(station_map_path)
df_weather = pd.read_csv(weather_path)

def strip_kv_suffix(station: str) -> str:
    if pd.isna(station):
        return station
    s = str(station).strip()
    parts = s.split()
    if len(parts) >= 2 and ("kv" in parts[-1].lower()):  # e.g. 66_11kV / 132_11kV
        return " ".join(parts[:-1]).strip()
    return s

# ① 映射表生成 station_clean（去掉 66_11kV 这种后缀）
df_map["station_clean"] = df_map["station"].apply(strip_kv_suffix)

# ② 自检：清洗后是否出现重复（会导致一个 station_clean 对应多个 station_id）
dup = df_map[df_map.duplicated("station_clean", keep=False)].sort_values("station_clean")
if len(dup) > 0:
    print("⚠️ 清洗后 station_clean 出现重复（一个名字对应多个 station_id），需要先处理：")
    print(dup[["station", "station_clean", "station_id"]].head(50))

# ③ 构建 station_clean -> station_id 映射
station2id = dict(zip(df_map["station_clean"], df_map["station_id"]))

# ④ 给天气表加 station_id
df_weather["station_id"] = df_weather["station_clean"].map(station2id)

# ⑤ 自检：是否有没映射上的站点
missing = df_weather[df_weather["station_id"].isna()]["station_clean"].dropna().unique()
if len(missing) > 0:
    print("⚠️ 以下天气站点在映射表中找不到（station_clean 不匹配）：")
    print(missing)
else:
    print("✅ 所有天气站点均成功映射 station_id")

df_weather.to_csv(output_path, index=False)
print("Saved:", output_path)
