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
STATIONS_LIMIT = None  # 例如 5


# =========================
# 工具函数
# =========================
def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """清理列名首尾空格"""
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_station_name(name: str) -> str:
    """
    站点名归一化：
    例如 'Aberdeen 66_11kV' -> 'Aberdeen'
    """
    if pd.isna(name):
        return None
    name = str(name).strip()

    # 去掉类似 " 66_11kV" / " 132_33kV" / " 11kV" 这一类尾缀
    name = re.sub(r"\s+\d+(?:_\d+)?kV\b", "", name, flags=re.IGNORECASE)

    # 顺手压缩多余空格
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
    """
    从 station_map.csv 建立:
        归一化后的 station_name -> station_id
    """
    station_map_df = normalize_colnames(station_map_df)

    # 兼容列名可能有空格或大小写问题
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
    """
    获取单个站点 hourly 天气数据
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join([
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "shortwave_radiation",
        ]),
        "timezone": TIMEZONE,
    }

    data = safe_request(params)

    hourly = data["hourly"]

    # API 正常时这些数组长度应一致
    times = hourly.get("time", [])
    temp = hourly.get("temperature_2m", [])
    precip = hourly.get("precipitation", [])
    wind = hourly.get("wind_speed_10m", [])
    radiation = hourly.get("shortwave_radiation", [])

    n = len(times)
    if not (len(temp) == len(precip) == len(wind) == len(radiation) == n):
        raise ValueError(
            f"{station_name} 返回字段长度不一致: "
            f"time={len(times)}, temp={len(temp)}, precip={len(precip)}, "
            f"wind={len(wind)}, radiation={len(radiation)}"
        )

    # 按你的目标格式输出
    df = pd.DataFrame({
        "date": times,
        "temperature_2m_mean": temp,
        "temperature_2m_max": temp,
        "temperature_2m_min": temp,
        "precipitation_sum": precip,
        "wind_speed_10m_mean": wind,
        "shortwave_radiation_sum": radiation,
        "station_clean": station_name,
        "lat": lat,
        "lon": lon,
    })

    return df


# =========================
# 主流程
# =========================
def main():
    # 1) 读取站点经纬度
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

    # 去重，避免重复站点
    stations_df = stations_df.drop_duplicates(subset=["station_clean"]).reset_index(drop=True)

    # 2) 读取 station -> station_id 映射
    station_map_df = pd.read_csv(STATION_MAP_FILE)
    station_id_map = build_station_id_map(station_map_df)

    # 3) 给站点表补 station_id
    stations_df["station_id"] = stations_df["station_clean"].map(station_id_map)

    # 打印映射缺失情况
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

    # 列顺序固定成你要的格式
    result = result[
        [
            "date",
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_mean",
            "shortwave_radiation_sum",
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