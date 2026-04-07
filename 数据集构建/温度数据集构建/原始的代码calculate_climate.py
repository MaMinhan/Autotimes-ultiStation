import pandas as pd
import requests
import time
import os
from tqdm import tqdm

IN_CSV = "stations_clean.csv"
OUT_CSV = "station_weather_daily_2023.csv"
FAIL_CSV = "station_weather_failures_2023.csv"

# ===== 只采集 2024 年 =====
START = "2023-01-01"
END   = "2023-12-31"

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_mean",
    "shortwave_radiation_sum",
]

# ===== 限流与重试参数（关键）=====
BASE_SLEEP = 1.2     # 每个请求后至少等待
MAX_RETRY = 5
BACKOFF_START = 2
TIMEOUT = 60
# =================================

def safe_get(url, params):
    backoff = BACKOFF_START
    for _ in range(MAX_RETRY):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)

            if r.status_code == 429:
                time.sleep(backoff)
                backoff *= 2
                continue

            r.raise_for_status()
            return r.json()

        except requests.exceptions.ReadTimeout:
            time.sleep(backoff)
            backoff *= 2
        except requests.exceptions.RequestException:
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError("Request failed after retries")

def fetch_station_daily(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START,
        "end_date": END,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Australia/Sydney",
    }
    js = safe_get(BASE_URL, params)

    if "daily" not in js or "time" not in js["daily"]:
        return pd.DataFrame()

    daily = js["daily"]
    df = pd.DataFrame({"date": daily["time"]})
    for v in DAILY_VARS:
        df[v] = daily.get(v, [None] * len(df))

    return df

def main():
    stations = pd.read_csv(IN_CSV, skipinitialspace=True)

    # 1) 清理空坐标
    stations["lat"] = pd.to_numeric(stations["lat"], errors="coerce")
    stations["lon"] = pd.to_numeric(stations["lon"], errors="coerce")
    stations = stations.dropna(subset=["lat", "lon"]).copy()

    # 2) 断点续跑
    done_stations = set()
    if os.path.exists(OUT_CSV):
        try:
            done = pd.read_csv(OUT_CSV, usecols=["station_clean"])
            done_stations = set(done["station_clean"].astype(str).unique())
            print(f"🔁 已完成站点数: {len(done_stations)}")
        except Exception:
            pass

    buffer = []
    fails = []

    for _, row in tqdm(stations.iterrows(), total=len(stations)):
        name = str(row["station_clean"]).strip()
        lat = float(row["lat"])
        lon = float(row["lon"])

        if name in done_stations:
            continue

        try:
            df = fetch_station_daily(lat, lon)
            if df.empty:
                fails.append({"station_clean": name, "lat": lat, "lon": lon, "error": "empty"})
                continue

            df["station_clean"] = name
            df["lat"] = lat
            df["lon"] = lon

            buffer.append(df)

            # 每 5 个站点就落盘一次（防中断）
            if len(buffer) >= 5:
                out_part = pd.concat(buffer, ignore_index=True)
                out_part.to_csv(
                    OUT_CSV,
                    index=False,
                    mode="a",
                    header=not os.path.exists(OUT_CSV),
                    encoding="utf-8-sig",
                )
                buffer = []

            time.sleep(BASE_SLEEP)

        except Exception as e:
            fails.append({"station_clean": name, "lat": lat, "lon": lon, "error": repr(e)})
            time.sleep(BASE_SLEEP)

    # flush
    if buffer:
        out_part = pd.concat(buffer, ignore_index=True)
        out_part.to_csv(
            OUT_CSV,
            index=False,
            mode="a",
            header=not os.path.exists(OUT_CSV),
            encoding="utf-8-sig",
        )

    if fails:
        pd.DataFrame(fails).to_csv(FAIL_CSV, index=False, encoding="utf-8-sig")
        print("⚠️ 失败清单:", FAIL_CSV)

    print("✅ 2023 年气象数据采集完成:", OUT_CSV)

if __name__ == "__main__":
    main()
