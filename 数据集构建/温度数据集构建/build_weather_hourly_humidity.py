import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm

# =========================
# 配置
# =========================
STATIONS_FILE = "/root/autodl-tmp/datasets/SelfMadeAusgridData/stations_to_SA2_SA3_SA4_2021.csv"
STATION_MAP_FILE = "/root/autodl-tmp/datasets/SelfMadeAusgridData/station_map.csv"
OUTPUT_FILE = "/root/autodl-tmp/datasets/SelfMadeAusgridData/weather_hourly_20210501_20240430.csv"

START_DATE = "2021-05-01"
END_DATE = "2024-04-30"

API_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEZONE = "Australia/Sydney"

# 小样本调试开关；不需要可设为 None
STATIONS_LIMIT = None

# =========================
# 推荐保留的 hourly 变量
# =========================
HOURLY_VARS = [
    "temperature_2m",          # 温度
    "relative_humidity_2m",    # 湿度
    "apparent_temperature",    # 体感温度
    "precipitation",           # 降水
    "wind_speed_10m",          # 风速
    "wind_gusts_10m",          # 阵风（可选但一起拿下来）
    "shortwave_radiation",     # 短波辐射
    "cloud_cover",             # 云量
]

# =========================
# 工具函数
# =========================
def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_station_name(name: str) -> str:
    if pd.isna(name):
        return None
    name = str(name).strip()
    name = re.sub(r"\s+\d+(?:_\d+)?kV\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def safe_request(params: dict, max_retries: int = 5, sleep_sec: float = 1.5):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(API_URL, params=params, timeout=120)
            r.raise_for_status()
            data = r.json()

            if "hourly" not in data or data["hourly"] is None:
                raise ValueError(f"API返回中没有 hourly 字段: {data}")

            return data
        except Exception as e:
            last_err = e
            print(f"[WARN] 请求失败，第 {attempt}/{max_retries} 次：{e}")
            time.sleep(sleep_sec * attempt)

    raise RuntimeError(f"请求最终失败：{last_err}")


def build_station_id_map(station_map_df: pd.DataFrame) -> dict:
    station_map_df = normalize_colnames(station_map_df)

    col_station = None
    col_station_id = None

    for c in station_map_df.columns:
        cl = c.lower()
        if cl == "station":
            col_station = c
        elif cl == "station_id":
            col_station_id = c

    if col_station is None or col_station_id is None:
        raise ValueError(
            f"station_map.csv 必须包含 station 和 station_id 两列，当前列为: {station_map_df.columns.tolist()}"
        )

    mapping = {}
    for _, row in station_map_df.iterrows():
        raw_name = row[col_station]
        sid = row[col_station_id]
        norm_name = normalize_station_name(raw_name)
        if norm_name is not None and norm_name != "":
            mapping[norm_name] = sid

    return mapping


def fetch_hourly_weather_for_station(station_name: str, lat: float, lon: float) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    }

    data = safe_request(params)
    hourly = data["hourly"]

    times = hourly.get("time", [])
    temp = hourly.get("temperature_2m", [])
    rh = hourly.get("relative_humidity_2m", [])
    apparent_temp = hourly.get("apparent_temperature", [])
    precip = hourly.get("precipitation", [])
    wind = hourly.get("wind_speed_10m", [])
    wind_gust = hourly.get("wind_gusts_10m", [])
    radiation = hourly.get("shortwave_radiation", [])
    cloud_cover = hourly.get("cloud_cover", [])

    n = len(times)
    fields = {
        "temperature_2m": temp,
        "relative_humidity_2m": rh,
        "apparent_temperature": apparent_temp,
        "precipitation": precip,
        "wind_speed_10m": wind,
        "wind_gusts_10m": wind_gust,
        "shortwave_radiation": radiation,
        "cloud_cover": cloud_cover,
    }

    for k, v in fields.items():
        if len(v) != n:
            raise ValueError(
                f"{station_name} 返回字段长度不一致: time={n}, {k}={len(v)}"
            )

    df = pd.DataFrame({
        "date": times,

        # 温度
        "temperature_2m_mean": temp,
        "temperature_2m_max": temp,
        "temperature_2m_min": temp,

        # 湿度
        "relative_humidity_2m_mean": rh,
        "relative_humidity_2m_max": rh,
        "relative_humidity_2m_min": rh,

        # 体感温度
        "apparent_temperature_mean": apparent_temp,
        "apparent_temperature_max": apparent_temp,
        "apparent_temperature_min": apparent_temp,

        # 其他气象量
        "precipitation_sum": precip,
        "wind_speed_10m_mean": wind,
        "wind_gusts_10m_max": wind_gust,
        "shortwave_radiation_sum": radiation,
        "cloud_cover_mean": cloud_cover,

        "station_clean": station_name,
        "lat": lat,
        "lon": lon,
    })

    return df


# =========================
# 主流程
# =========================
def main():
    stations_df = pd.read_csv(STATIONS_FILE)
    stations_df = normalize_colnames(stations_df)

    required_cols = ["station_clean", "lat", "lon"]
    for c in required_cols:
        if c not in stations_df.columns:
            raise ValueError(
                f"stations_to_SA2_SA3_SA4_2021.csv 缺少列 {c}，当前列为: {stations_df.columns.tolist()}"
            )

    stations_df = stations_df[["station_clean", "lat", "lon"]].copy()
    stations_df["station_clean"] = stations_df["station_clean"].astype(str).str.strip()
    stations_df = stations_df.drop_duplicates(subset=["station_clean"]).reset_index(drop=True)

    station_map_df = pd.read_csv(STATION_MAP_FILE)
    station_id_map = build_station_id_map(station_map_df)

    stations_df["station_id"] = stations_df["station_clean"].map(station_id_map)

    missing_map = stations_df[stations_df["station_id"].isna()]
    if len(missing_map) > 0:
        print(f"[WARN] 有 {len(missing_map)} 个站点没有匹配到 station_id：")
        print(missing_map.head(20).to_string(index=False))

    if STATIONS_LIMIT is not None:
        stations_df = stations_df.head(STATIONS_LIMIT).copy()
        print(f"[INFO] 启用 STATIONS_LIMIT={STATIONS_LIMIT}，当前只处理前 {len(stations_df)} 个站点")

    print(f"[INFO] 总站点数: {len(stations_df)}")
    print(f"[INFO] 日期范围: {START_DATE} ~ {END_DATE}")
    print(f"[INFO] 输出文件: {OUTPUT_FILE}")
    print(f"[INFO] hourly vars: {HOURLY_VARS}")

    all_parts = []
    failed_stations = []

    for _, row in tqdm(stations_df.iterrows(), total=len(stations_df), desc="Fetching weather"):
        station_name = row["station_clean"]
        lat = row["lat"]
        lon = row["lon"]
        station_id = row["station_id"]

        try:
            part = fetch_hourly_weather_for_station(station_name, lat, lon)
            part["station_id"] = station_id
            all_parts.append(part)
        except Exception as e:
            print(f"[ERROR] 站点 {station_name} 获取失败: {e}")
            failed_stations.append({
                "station_clean": station_name,
                "lat": lat,
                "lon": lon,
                "station_id": station_id,
                "error": str(e),
            })

    if len(all_parts) == 0:
        raise RuntimeError("没有任何站点成功获取数据，未生成输出文件。")

    result = pd.concat(all_parts, ignore_index=True)

    result = result[
        [
            "date",
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "relative_humidity_2m_mean",
            "relative_humidity_2m_max",
            "relative_humidity_2m_min",
            "apparent_temperature_mean",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "precipitation_sum",
            "wind_speed_10m_mean",
            "wind_gusts_10m_max",
            "shortwave_radiation_sum",
            "cloud_cover_mean",
            "station_clean",
            "lat",
            "lon",
            "station_id",
        ]
    ]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\n[INFO] 保存完成")
    print(f"[INFO] 输出行数: {len(result)}")
    print(f"[INFO] 输出路径: {OUTPUT_FILE}")
    print("[INFO] 结果预览:")
    print(result.head(10).to_string(index=False))

    if failed_stations:
        failed_df = pd.DataFrame(failed_stations)
        failed_path = OUTPUT_FILE.replace(".csv", "_failed_stations.csv")
        failed_df.to_csv(failed_path, index=False, encoding="utf-8")
        print(f"\n[WARN] 有失败站点，已保存到: {failed_path}")


if __name__ == "__main__":
    main()