import os
import re
import time
import random
import requests
import pandas as pd
from tqdm import tqdm

# =========================
# 配置
# =========================
STATIONS_FILE = "/root/autodl-tmp/datasets/SelfMadeAusgridData/stations_to_SA2_SA3_SA4_2021.csv"
STATION_MAP_FILE = "/root/autodl-tmp/datasets/SelfMadeAusgridData/station_map.csv"
OUTPUT_FILE = "/root/autodl-tmp/datasets/SelfMadeAusgridData/weather/weather_hourly_20210501_20240430_humid.csv"

START_DATE = "2021-05-01"
END_DATE = "2024-04-30"

API_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEZONE = "Australia/Sydney"

# 小样本调试开关；不需要可设为 None
STATIONS_LIMIT = None

# 断点续跑/节流配置
RESUME = True
SAVE_EVERY = 5                  # 每处理多少个成功站点落盘一次
BASE_SLEEP_SEC = 1.5            # 每个站点请求成功后基础等待
JITTER_SEC = 0.8                # 随机抖动，避免请求过于规律
MAX_RETRIES = 8                 # 最大重试次数
HTTP_429_BASE_WAIT = 15         # 429 首次等待秒数，后续指数退避
GENERAL_ERROR_BASE_WAIT = 3     # 非 429 错误首次等待秒数

# =========================
# 推荐保留的 hourly 变量
# =========================
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "cloud_cover",
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


def build_station_id_map(station_map_df: pd.DataFrame) -> dict:
    """
    从 station_map.csv 建立:
        归一化后的 station_name -> station_id

    注意：
    这种方式对重名站点（去掉电压等级后重名）有覆盖风险。
    这里先保留你的原逻辑，只是加 warning。
    """
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
    dup_names = {}

    for _, row in station_map_df.iterrows():
        raw_name = row[col_station]
        sid = row[col_station_id]
        norm_name = normalize_station_name(raw_name)

        if norm_name is not None and norm_name != "":
            if norm_name in mapping and mapping[norm_name] != sid:
                dup_names.setdefault(norm_name, set()).update([mapping[norm_name], sid])
            mapping[norm_name] = sid

    if dup_names:
        print("[WARN] 存在归一化后重名站点，station_id 可能发生覆盖：")
        for k, v in list(dup_names.items())[:20]:
            print(f"  {k}: {sorted(v)}")

    return mapping


def safe_request(params: dict, max_retries: int = MAX_RETRIES):
    """
    改进版请求：
    - 429: 单独指数退避
    - 其他错误: 普通退避
    - 支持读取 Retry-After
    """
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(API_URL, params=params, timeout=120)

            # 单独处理 429
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        wait_sec = max(float(retry_after), HTTP_429_BASE_WAIT)
                    except Exception:
                        wait_sec = HTTP_429_BASE_WAIT * (2 ** (attempt - 1))
                else:
                    wait_sec = HTTP_429_BASE_WAIT * (2 ** (attempt - 1))

                wait_sec += random.uniform(0, 1.5)
                print(f"[WARN] 请求失败，第 {attempt}/{max_retries} 次：429 Too Many Requests，等待 {wait_sec:.1f}s 后重试")
                time.sleep(wait_sec)
                continue

            r.raise_for_status()
            data = r.json()

            if "hourly" not in data or data["hourly"] is None:
                raise ValueError(f"API返回中没有 hourly 字段: {data}")

            return data

        except Exception as e:
            last_err = e

            # 非 429 的普通退避
            wait_sec = GENERAL_ERROR_BASE_WAIT * attempt + random.uniform(0, 1.0)
            print(f"[WARN] 请求失败，第 {attempt}/{max_retries} 次：{e}，等待 {wait_sec:.1f}s 后重试")
            time.sleep(wait_sec)

    raise RuntimeError(f"请求最终失败：{last_err}")


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

        "temperature_2m_mean": temp,
        "temperature_2m_max": temp,
        "temperature_2m_min": temp,

        "relative_humidity_2m_mean": rh,
        "relative_humidity_2m_max": rh,
        "relative_humidity_2m_min": rh,

        "apparent_temperature_mean": apparent_temp,
        "apparent_temperature_max": apparent_temp,
        "apparent_temperature_min": apparent_temp,

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


def get_result_columns():
    return [
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


def save_parts_to_csv(existing_df: pd.DataFrame, new_parts: list, output_file: str):
    cols = get_result_columns()

    frames = []
    if existing_df is not None and len(existing_df) > 0:
        frames.append(existing_df)

    if new_parts:
        new_df = pd.concat(new_parts, ignore_index=True)
        new_df = new_df[cols]
        frames.append(new_df)

    if not frames:
        return

    result = pd.concat(frames, ignore_index=True)

    # 去重：同一 station_id + date 保留最后一条
    result = result.drop_duplicates(subset=["station_id", "date"], keep="last")
    result = result[cols]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result.to_csv(output_file, index=False, encoding="utf-8")


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
    print(f"[INFO] RESUME={RESUME}")

    # -------------------------
    # 断点续跑：读取已有结果
    # -------------------------
    existing_df = None
    done_station_ids = set()

    if RESUME and os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE, usecols=["station_id", "date"] + [c for c in get_result_columns() if c not in ["station_id", "date"]])
            existing_df["station_id"] = pd.to_numeric(existing_df["station_id"], errors="coerce")
            existing_df = existing_df.dropna(subset=["station_id", "date"]).copy()
            existing_df["station_id"] = existing_df["station_id"].astype(int)

            # 只要某站点已有数据，就先视为完成，避免重复请求
            done_station_ids = set(existing_df["station_id"].unique().tolist())
            print(f"[RESUME] 检测到已有输出文件，已完成站点数: {len(done_station_ids)}")
        except Exception as e:
            print(f"[WARN] 读取已有输出文件失败，将从头开始追加新结果: {e}")
            existing_df = None
            done_station_ids = set()

    # 待处理站点
    to_process = stations_df.copy()
    if RESUME and done_station_ids:
        to_process = to_process[~to_process["station_id"].isin(done_station_ids)].reset_index(drop=True)

    print(f"[INFO] 待处理站点数: {len(to_process)}")

    all_new_parts = []
    failed_stations = []

    success_count = 0

    for _, row in tqdm(to_process.iterrows(), total=len(to_process), desc="Fetching weather"):
        station_name = row["station_clean"]
        lat = row["lat"]
        lon = row["lon"]
        station_id = row["station_id"]

        try:
            part = fetch_hourly_weather_for_station(station_name, lat, lon)
            part["station_id"] = station_id
            all_new_parts.append(part)

            success_count += 1

            # 周期性落盘，防止中途断掉
            if success_count % SAVE_EVERY == 0:
                save_parts_to_csv(existing_df, all_new_parts, OUTPUT_FILE)

                # 落盘后把 new_parts 并入 existing_df，清空缓存
                if existing_df is None:
                    existing_df = pd.concat(all_new_parts, ignore_index=True)
                else:
                    existing_df = pd.concat([existing_df, pd.concat(all_new_parts, ignore_index=True)], ignore_index=True)
                    existing_df = existing_df.drop_duplicates(subset=["station_id", "date"], keep="last")

                all_new_parts = []
                print(f"[CHECKPOINT] 已阶段性保存，success_count={success_count}")

            # 成功后主动限速
            sleep_sec = BASE_SLEEP_SEC + random.uniform(0, JITTER_SEC)
            time.sleep(sleep_sec)

        except Exception as e:
            print(f"[ERROR] 站点 {station_name} 获取失败: {e}")
            failed_stations.append({
                "station_clean": station_name,
                "lat": lat,
                "lon": lon,
                "station_id": station_id,
                "error": str(e),
            })

            # 失败后也稍等一下，避免连续失败进一步触发限流
            time.sleep(3.0 + random.uniform(0, 1.0))

    # 收尾落盘
    save_parts_to_csv(existing_df, all_new_parts, OUTPUT_FILE)

    # 读回最终结果统计
    if os.path.exists(OUTPUT_FILE):
        result = pd.read_csv(OUTPUT_FILE)
        print("\n[INFO] 保存完成")
        print(f"[INFO] 输出行数: {len(result)}")
        print(f"[INFO] 输出路径: {OUTPUT_FILE}")
        print("[INFO] 结果预览:")
        print(result.head(10).to_string(index=False))
        print(f"[INFO] 已写入 station_id 数: {result['station_id'].nunique() if 'station_id' in result.columns else 'N/A'}")
    else:
        raise RuntimeError("未生成输出文件。")

    if failed_stations:
        failed_df = pd.DataFrame(failed_stations)
        failed_path = OUTPUT_FILE.replace(".csv", "_failed_stations.csv")
        failed_df.to_csv(failed_path, index=False, encoding="utf-8")
        print(f"\n[WARN] 有失败站点，已保存到: {failed_path}")
        print(f"[WARN] 失败站点数: {len(failed_df)}")
    else:
        print("\n[INFO] 所有待处理站点均成功。")


if __name__ == "__main__":
    main()