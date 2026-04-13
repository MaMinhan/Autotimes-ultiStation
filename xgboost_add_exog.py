import os
import re
import glob
import argparse
import numpy as np
import pandas as pd

import re

def normalize_station_clean(name: str) -> str:
    """
    把 prediction parquet 里的站点名清洗成 station_clean，
    例如：
    'Aberdeen 66_11kV' -> 'Aberdeen'
    'Adamstown 132_11kV' -> 'Adamstown'
    """
    if pd.isna(name):
        return ""

    s = str(name).strip()

    # 去掉末尾类似 " 66_11kV" / " 132_11kV" / " 33_11kV"
    s = re.sub(r"\s+\d+_\d+kV$", "", s, flags=re.IGNORECASE)

    return s.strip()
# =========================================================
# 1) Holiday / Calendar prefix
# 与 data_loader.py 的 _build_calendar_prefix 对齐
# =========================================================
def load_holiday_dates(holiday_csv: str):
    if holiday_csv is None or (not os.path.exists(holiday_csv)):
        return set()

    df_h = pd.read_csv(holiday_csv)
    df_h["date"] = pd.to_datetime(df_h["date"]).dt.date
    df_h["is_holiday"] = (
        df_h["is_holiday"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    return set(df_h.loc[df_h["is_holiday"] == True, "date"].tolist())


def build_calendar_prefix(ts: pd.Timestamp, holiday_dates: set) -> dict:
    """
    与 data_loader.py 的 _build_calendar_prefix 保持一致
    输出 18 维:
    [month_norm] +
    [season_onehot(4)] +
    [dow_onehot(7)] +
    [day_type_onehot(3)] +
    [holiday_rel_onehot(3)]
    """
    ts = pd.Timestamp(ts)

    # 1) month_norm
    month_norm = ts.month / 12.0

    # 2) season onehot: [spring, summer, autumn, winter]
    season = [0.0, 0.0, 0.0, 0.0]
    if ts.month in [9, 10, 11]:
        season[0] = 1.0
    elif ts.month in [12, 1, 2]:
        season[1] = 1.0
    elif ts.month in [3, 4, 5]:
        season[2] = 1.0
    else:
        season[3] = 1.0

    # 3) day-of-week onehot
    dow = [0.0] * 7
    dow[ts.weekday()] = 1.0

    # 4) day_type onehot: [workday, weekend, holiday]
    d = ts.date()
    is_holiday = d in holiday_dates
    is_weekend = ts.weekday() >= 5
    is_workday = (not is_weekend) and (not is_holiday)

    day_type = [0.0, 0.0, 0.0]
    if is_workday:
        day_type[0] = 1.0
    elif is_weekend and (not is_holiday):
        day_type[1] = 1.0
    else:
        day_type[2] = 1.0

    # 5) holiday_rel onehot: [holiday-1, holiday, holiday+1]
    prev_day = (ts - pd.Timedelta(days=1)).date()
    next_day = (ts + pd.Timedelta(days=1)).date()

    holiday_rel = [0.0, 0.0, 0.0]
    if next_day in holiday_dates:
        holiday_rel[0] = 1.0   # 明天是节假日
    if d in holiday_dates:
        holiday_rel[1] = 1.0   # 今天是节假日
    if prev_day in holiday_dates:
        holiday_rel[2] = 1.0   # 昨天是节假日

    feat = {
        "prefix_cal_month_norm": float(month_norm),

        "prefix_cal_season_spring": float(season[0]),
        "prefix_cal_season_summer": float(season[1]),
        "prefix_cal_season_autumn": float(season[2]),
        "prefix_cal_season_winter": float(season[3]),

        "prefix_cal_dow_0": float(dow[0]),
        "prefix_cal_dow_1": float(dow[1]),
        "prefix_cal_dow_2": float(dow[2]),
        "prefix_cal_dow_3": float(dow[3]),
        "prefix_cal_dow_4": float(dow[4]),
        "prefix_cal_dow_5": float(dow[5]),
        "prefix_cal_dow_6": float(dow[6]),

        "prefix_cal_daytype_workday": float(day_type[0]),
        "prefix_cal_daytype_weekend": float(day_type[1]),
        "prefix_cal_daytype_holiday": float(day_type[2]),

        "prefix_cal_holidayrel_pre": float(holiday_rel[0]),
        "prefix_cal_holidayrel_cur": float(holiday_rel[1]),
        "prefix_cal_holidayrel_post": float(holiday_rel[2]),
    }
    return feat


# =========================================================
# 2) Social prefix
# 与 data_loader.py 的 social_prefix 逻辑对齐
# =========================================================
SOCIAL_COLS_CANONICAL = [
    "pop_density_per_sqkm_norm",
    "young_ratio_0_14_norm",
    "working_ratio_15_64_norm",
    "elderly_ratio_65_plus_norm",
    "private_dwelling_ratio_norm",
    "other_dwelling_ratio_norm",
]

SOCIAL_RENAME_MAP = {
    "pop_density_norm": "pop_density_per_sqkm_norm",
    "young_ratio_norm": "young_ratio_0_14_norm",
    "working_ratio_norm": "working_ratio_15_64_norm",
    "elderly_ratio_norm": "elderly_ratio_65_plus_norm",
}


def load_social_lookup(social_csv: str, station_sa2_map_csv: str = None):
    """
    支持三种 social 输入：
    A. stationname 级
       stationname + 6个 norm 列
    B. sid_idx 级
       sid_idx + 6个 norm 列
    C. SA2 级
       SA2_CODE21 + 6个 norm 列
       需要 station_sa2_map_csv 来做站点映射
    """
    if social_csv is None or (not os.path.exists(social_csv)):
        return {"mode": "none", "by_station": {}, "by_sid": {}}

    df_s = pd.read_csv(social_csv, skipinitialspace=True)
    df_s.columns = df_s.columns.str.strip()

    # 兼容你现有 social 文件列名
    for old, new in SOCIAL_RENAME_MAP.items():
        if old in df_s.columns and new not in df_s.columns:
            df_s = df_s.rename(columns={old: new})

    for col in SOCIAL_COLS_CANONICAL:
        if col in df_s.columns:
            df_s[col] = pd.to_numeric(df_s[col], errors="coerce")

    # ---------- A. stationname 级 ----------
    if "stationname" in df_s.columns:
        df_s["stationname"] = df_s["stationname"].astype(str).str.strip()

        by_station = {}
        for _, row in df_s.iterrows():
            station_name = row["stationname"]
            vec = {}
            for c in SOCIAL_COLS_CANONICAL:
                vec[f"prefix_social_{c}"] = float(row[c]) if c in df_s.columns and pd.notna(row[c]) else np.nan
            by_station[station_name] = vec

        return {"mode": "stationname", "by_station": by_station, "by_sid": {}}

    # ---------- B. sid_idx 级 ----------
    if "sid_idx" in df_s.columns:
        by_sid = {}
        for _, row in df_s.iterrows():
            sid = int(row["sid_idx"])
            vec = {}
            for c in SOCIAL_COLS_CANONICAL:
                vec[f"prefix_social_{c}"] = float(row[c]) if c in df_s.columns and pd.notna(row[c]) else np.nan
            by_sid[sid] = vec

        return {"mode": "sid_idx", "by_station": {}, "by_sid": by_sid}

    # ---------- C. SA2 级 ----------
    if "SA2_CODE21" in df_s.columns:
        if station_sa2_map_csv is None or (not os.path.exists(station_sa2_map_csv)):
            raise ValueError(
                "social_csv is SA2-level, so you must provide --station_sa2_map_csv"
            )

        df_map = pd.read_csv(station_sa2_map_csv, skipinitialspace=True)
        df_map.columns = df_map.columns.str.strip()

        # 你的映射表里是 station_clean
        if "station_clean" not in df_map.columns:
            raise ValueError("station_sa2_map_csv must contain 'station_clean' column")

        if "SA2_CODE21" not in df_map.columns:
            raise ValueError("station_sa2_map_csv must contain 'SA2_CODE21' column")

        df_map["station_clean"] = df_map["station_clean"].astype(str).str.strip()
        df_map["SA2_CODE21"] = pd.to_numeric(df_map["SA2_CODE21"], errors="coerce")
        df_s["SA2_CODE21"] = pd.to_numeric(df_s["SA2_CODE21"], errors="coerce")

        merged = df_map.merge(df_s, on="SA2_CODE21", how="left")

        by_station = {}
        for _, row in merged.iterrows():
            station_clean = str(row["station_clean"]).strip()

            vec = {}
            for c in SOCIAL_COLS_CANONICAL:
                vec[f"prefix_social_{c}"] = float(row[c]) if c in merged.columns and pd.notna(row[c]) else np.nan

            by_station[station_clean] = vec

        return {"mode": "sa2_map", "by_station": by_station, "by_sid": {}}

    raise ValueError(
        "social_csv format not recognized. Need one of:\n"
        "1) stationname + social cols\n"
        "2) sid_idx + social cols\n"
        "3) SA2_CODE21 + social cols, plus --station_sa2_map_csv"
    )


def get_social_prefix(station_name, sid_idx, social_lookup):
    if social_lookup["mode"] == "none":
        return {f"prefix_social_{c}": np.nan for c in SOCIAL_COLS_CANONICAL}

    # 原站点名
    station_name = str(station_name).strip() if station_name is not None else None

    # 清洗后的 station_clean
    station_clean = normalize_station_clean(station_name) if station_name is not None else None

    # 优先尝试原始 stationname
    if station_name is not None and station_name in social_lookup["by_station"]:
        return social_lookup["by_station"][station_name]

    # 再尝试清洗后的 station_clean
    if station_clean is not None and station_clean in social_lookup["by_station"]:
        return social_lookup["by_station"][station_clean]

    # 再尝试 sid_idx
    if sid_idx in social_lookup["by_sid"]:
        return social_lookup["by_sid"][sid_idx]

    return {f"prefix_social_{c}": np.nan for c in SOCIAL_COLS_CANONICAL}
# =========================================================
# 3) Temperature features
# 沿用你之前脚本的思路，但按 origin_time 算，避免泄露
# =========================================================
def load_temperature_map(temp_csv: str):
    if temp_csv is None or (not os.path.exists(temp_csv)):
        return {}

    df_t = pd.read_csv(temp_csv, skipinitialspace=True)
    df_t.columns = df_t.columns.str.strip()

    # 兼容你示例里的字段
    if "date" in df_t.columns and "datetime" not in df_t.columns:
        df_t["datetime"] = pd.to_datetime(df_t["date"], errors="coerce")
    elif "datetime" in df_t.columns:
        df_t["datetime"] = pd.to_datetime(df_t["datetime"], errors="coerce")
    else:
        raise ValueError("temp_csv needs 'date' or 'datetime' column.")

    if "station_id" in df_t.columns and "sid_idx" not in df_t.columns:
        df_t["sid_idx"] = pd.to_numeric(df_t["station_id"], errors="coerce")
    elif "sid_idx" in df_t.columns:
        df_t["sid_idx"] = pd.to_numeric(df_t["sid_idx"], errors="coerce")
    else:
        raise ValueError("temp_csv needs 'station_id' or 'sid_idx' column.")

    # 主温度列
    if "temperature_2m_mean" in df_t.columns and "temperature" not in df_t.columns:
        df_t["temperature"] = pd.to_numeric(df_t["temperature_2m_mean"], errors="coerce")
    elif "temperature" in df_t.columns:
        df_t["temperature"] = pd.to_numeric(df_t["temperature"], errors="coerce")
    else:
        raise ValueError("temp_csv needs 'temperature_2m_mean' or 'temperature' column.")

    df_t = df_t.dropna(subset=["datetime", "sid_idx"]).copy()
    df_t["sid_idx"] = df_t["sid_idx"].astype(int)

    temp_map = {
        sid: g[["datetime", "temperature"]].sort_values("datetime").reset_index(drop=True)
        for sid, g in df_t.groupby("sid_idx")
    }
    return temp_map


def safe_temperature_features(temp_df: pd.DataFrame, origin_time: pd.Timestamp):
    if temp_df is None or len(temp_df) == 0:
        return {
            "temp_last": np.nan,
            "temp_mean_4": np.nan,
            "temp_mean_96": np.nan,
            "temp_std_96": np.nan,
        }

    past = temp_df[temp_df["datetime"] < origin_time].sort_values("datetime")
    x = past["temperature"].to_numpy(dtype=float)

    def last():
        return float(x[-1]) if len(x) >= 1 else np.nan

    def roll_mean(win):
        return float(np.mean(x[-win:])) if len(x) >= win else np.nan

    def roll_std(win):
        return float(np.std(x[-win:])) if len(x) >= win else np.nan

    return {
        "temp_last": last(),
        "temp_mean_4": roll_mean(4),
        "temp_mean_96": roll_mean(96),
        "temp_std_96": roll_std(96),
    }


# =========================================================
# 4) 主处理逻辑：逐个 parquet part 追加外生变量
# =========================================================
def process_one_part(df, holiday_dates, social_lookup, temp_map, freq_minutes):
    required = ["sid_idx", "target_time", "horizon"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column in parquet: {c}")

    df = df.copy()
    df["target_time"] = pd.to_datetime(df["target_time"], errors="coerce")
    df["sid_idx"] = pd.to_numeric(df["sid_idx"], errors="coerce").astype("Int64")
    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").astype("Int64")

    has_station_name = "station_name" in df.columns
    if not has_station_name:
        df["station_name"] = ""

    delta = pd.Timedelta(minutes=freq_minutes)

    # 与 data_loader.py 对齐：prefix 的时间点是 forecast origin
    df["origin_time"] = df["target_time"] - (df["horizon"] - 1).astype("int64") * delta
    df["origin_time"] = pd.to_datetime(df["origin_time"], errors="coerce")

    # ---------------------------------------------------------
    # 1) 只对唯一 origin 计算一次特征
    # ---------------------------------------------------------
    origin_df = (
        df[["sid_idx", "station_name", "origin_time"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    feature_rows = []

    for row in origin_df.itertuples(index=False):
        sid_idx = int(row.sid_idx)
        station_name = row.station_name
        origin_time = pd.Timestamp(row.origin_time)

        feat = {
            "sid_idx": sid_idx,
            "station_name": station_name,
            "origin_time": origin_time,
        }

        # 1) holiday/calendar prefix —— 按 origin_time
        feat.update(build_calendar_prefix(origin_time, holiday_dates))

        # 2) social prefix —— 按站点
        feat.update(get_social_prefix(station_name, sid_idx, social_lookup))

        # 3) 温度统计 —— 按 origin_time
        temp_df = temp_map.get(sid_idx, None)
        feat.update(safe_temperature_features(temp_df, origin_time))

        feature_rows.append(feat)

    origin_feat_df = pd.DataFrame(feature_rows)

    # ---------------------------------------------------------
    # 2) merge 回原始 parquet
    # ---------------------------------------------------------
    df = df.merge(
        origin_feat_df,
        on=["sid_idx", "station_name", "origin_time"],
        how="left"
    )

    return df

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_dir", type=str, required=True, help="原始 prediction parquet 目录")
    parser.add_argument("--holiday_csv", type=str, required=True, help="holiday.csv")
    parser.add_argument("--social_csv", type=str, required=True, help="social csv")
    parser.add_argument("--temp_csv", type=str, required=True, help="temperature csv")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录（不会覆盖原 parquet）")
    parser.add_argument(
        "--station_sa2_map_csv",
        type=str,
        default="",
        help="站点到 SA2 的映射表，例如 stations_to_SA2_SA3_SA4_2021.csv"
    )
    parser.add_argument("--freq_minutes", type=int, default=15)
    parser.add_argument(
        "--start_part_idx",
        type=int,
        default=None,
        help="从第几个 parquet part 开始处理，例如 100 表示从 part_00100.parquet 开始"
    )
    parser.add_argument(
        "--start_part_name",
        type=str,
        default="",
        help="从指定 parquet 文件名开始处理，例如 part_00100.parquet"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 读 lookup
    print("[1/4] loading holiday...")
    holiday_dates = load_holiday_dates(args.holiday_csv)

    print("[2/4] loading social...")
    social_lookup = load_social_lookup(
        args.social_csv,
        station_sa2_map_csv=args.station_sa2_map_csv if args.station_sa2_map_csv else None
    )
    print("[3/4] loading temperature...")
    temp_map = load_temperature_map(args.temp_csv)

    # 逐 part 处理
    print("[4/4] processing parquet parts...")
    part_files = sorted(glob.glob(os.path.join(args.pred_dir, "part_*.parquet")))
    if not part_files:
        raise ValueError(f"No part_*.parquet found in {args.pred_dir}")

    # 支持从指定 part 文件名开始
    if args.start_part_name:
        base_names = [os.path.basename(x) for x in part_files]
        if args.start_part_name not in base_names:
            raise ValueError(
                f"start_part_name={args.start_part_name} not found in {args.pred_dir}\n"
                f"available examples: {base_names[:5]}"
            )
        start_pos = base_names.index(args.start_part_name)
        part_files = part_files[start_pos:]
        print(f"[RESUME] start_part_name={args.start_part_name}, start_pos={start_pos}")

    # 支持从指定 part 序号开始
    elif args.start_part_idx is not None:
        target_name = f"part_{args.start_part_idx:05d}.parquet"
        base_names = [os.path.basename(x) for x in part_files]
        if target_name not in base_names:
            raise ValueError(
                f"target part not found: {target_name} in {args.pred_dir}"
            )
        start_pos = base_names.index(target_name)
        part_files = part_files[start_pos:]
        print(f"[RESUME] start_part_idx={args.start_part_idx}, target_name={target_name}, start_pos={start_pos}")

    print(f"[PROCESS] total parts to process = {len(part_files)}")

    for i, fp in enumerate(part_files, 1):
        print(f"  [{i}/{len(part_files)}] processing {os.path.basename(fp)}")
        df = pd.read_parquet(fp)
        out_df = process_one_part(
            df=df,
            holiday_dates=holiday_dates,
            social_lookup=social_lookup,
            temp_map=temp_map,
            freq_minutes=args.freq_minutes
        )

        out_path = os.path.join(args.out_dir, os.path.basename(fp))
        out_df.to_parquet(out_path, index=False)

    # 把 _meta.parquet 一起复制过去（如果有）
    meta_fp = os.path.join(args.pred_dir, "_meta.parquet")
    if os.path.exists(meta_fp):
        meta_out = os.path.join(args.out_dir, "_meta.parquet")
        pd.read_parquet(meta_fp).to_parquet(meta_out, index=False)

    print("done:", args.out_dir)


if __name__ == "__main__":
    main()
    '''python xgboost_add_exog.py \
  --pred_dir /root/autodl-tmp/xgb_exports/CKPT_From_多尺度_time-pt_no_prefix/no_exog/val_predictions \
  --holiday_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/节假日/holiday.csv \
  --social_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/social/social_prefix_norm.csv \
  --station_sa2_map_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/stations_to_SA2_SA3_SA4_2021.csv \
  --temp_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/weather/weather_hourly_20210501_20240430.csv \
  --out_dir /root/autodl-tmp/xgb_exports/CKPT_From_多尺度_time-pt_no_prefix/with_exog/val \
  --freq_minutes 15 \
  --start_part_idx 0'''