import pandas as pd
import numpy as np
import os

SRC = "/root/autodl-tmp/datasets/merged_FY22_24_long.csv"                # 你的原始数据
OUT = "/root/autodl-tmp/datasets/merged_clean.csv"
MAP_OUT = "/root/autodl-tmp/datasets/station_map.csv"

FREQ = "15min"                 # 固定 15min
INTERP_LIMIT = 96              # 允许插值的最大连续缺失点数：96=1天(96*15min)

def main():
    df = pd.read_csv(SRC, usecols=["datetime", "station", "target"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["station"] = df["station"].astype(str)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")

    # 1) 基础清洗：去掉 datetime/station 为空的行；target 允许为空（后面补）
    df = df.dropna(subset=["datetime", "station"]).copy()
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)

    # 2) station_id 稳定编码（按 station 字典序）
    stations = sorted(df["station"].unique().tolist())
    station2id = {s: i for i, s in enumerate(stations)}
    df["station_id"] = df["station"].map(station2id).astype(int)

    pd.DataFrame({"station": stations, "station_id": list(range(len(stations)))}) \
      .to_csv(MAP_OUT, index=False)

    # 3) 构建全局时间轴（min~max，15min 网格）

    # Use the whole dataframe's datetime range (not `g` which is per-station)
    s_min = df["datetime"].min()
    s_max = df["datetime"].max()
    full_time = pd.date_range(s_min, s_max, freq=FREQ)

    # 4) 对每个 station 补齐网格 + 缺失 mask + 填充
    out_parts = []
    for s, g in df.groupby("station", sort=False):
        valid_cnt = g["target"].notna().sum()
        if valid_cnt < 672:
            print(f"skip station {s} with too few valid points: {valid_cnt}")
            continue
        g = g[["datetime", "station", "station_id", "target"]].copy()

        # 若同一 (station, datetime) 有重复，先聚合（建议取均值或最后一条）
        g = g.groupby(["datetime", "station", "station_id"], as_index=False)["target"].mean()

        g = g.set_index("datetime").reindex(full_time)
        g.index.name = "datetime"

        # station 信息补回去（reindex 后会变 NaN）
        g["station"] = s
        g["station_id"] = station2id[s]

        # miss mask：原始缺失为 1，原始有值为 0
        g["miss_target"] = g["target"].isna().astype(np.int8)

        # 5) 缺失填充（推荐策略）
        # 5.1 短缺口：时间插值（限制最大连续缺失长度，避免跨超长缺口硬插）
        g["target"] = g["target"].interpolate(
            method="time",
            limit=INTERP_LIMIT,
            limit_direction="both"
        )

        # 5.2 边界：ffill/bfill
        g["target"] = g["target"].ffill().bfill()

        # 5.3 兜底：仍有 NaN（极端情况）用该站点中位数或 0
        med = np.nanmedian(g["target"].values)
        if np.isnan(med):
            med = 0.0
        g["target"] = g["target"].fillna(med)

        out_parts.append(g.reset_index())

    out = pd.concat(out_parts, ignore_index=True)
    out = out.sort_values(["station_id", "datetime"]).reset_index(drop=True)

    # 先按天统计原始存在情况（用于后续判断和输出）
    out["date"] = out["datetime"].dt.floor("D")
    present_by_date = out.groupby("date")["miss_target"].apply(lambda x: (x == 0).sum())
    empty_dates = present_by_date[present_by_date == 0].index
    if len(empty_dates) > 0:
        print("Dates with no original data across all stations:")
        for d in empty_dates:
            print(d.strftime("%Y-%m-%d"))
        pd.Series([d.strftime("%Y-%m-%d") for d in empty_dates], name="date") \
          .to_csv('/root/autodl-tmp/datasets/all_empty_days.csv', index=False)
        print("saved: /root/autodl-tmp/datasets/all_empty_days.csv")
    else:
        print("No full-empty dates found.")

    # 按日期过滤：优先读取按日汇总文件，删除 pct_stations_missing == 1.0 的日期
    DROP_CSV = '/root/autodl-tmp/datasets/drop_dates.csv'
    DROP_TXT = '/root/autodl-tmp/datasets/drop_dates.txt'
    DATE_SUMMARY = '/root/autodl-tmp/datasets/missing_by_date_summary.csv'

    drop_dates = []
    if os.path.exists(DATE_SUMMARY):
        try:
            ddf = pd.read_csv(DATE_SUMMARY, parse_dates=['date'])
            if 'pct_stations_missing' in ddf.columns:
                dd = ddf[ddf['pct_stations_missing'] == 1.0]['date']
                drop_dates = [d.date() for d in pd.to_datetime(dd)]
                print(f'Loaded {len(drop_dates)} drop dates from {DATE_SUMMARY} where pct_stations_missing==1.0')
        except Exception as e:
            print('Failed reading', DATE_SUMMARY, e)

    # 回退：若没有从汇总文件获得 drop_dates，则使用完全为空的日期
    if len(drop_dates) == 0 and len(empty_dates) > 0:
        drop_dates = [pd.to_datetime(d).date() for d in empty_dates]
        print(f'Using {len(drop_dates)} empty_dates computed earlier')

    # 写出并应用过滤（如果有要删的日期）
    if len(drop_dates) > 0:
        pd.Series([d.strftime('%Y-%m-%d') for d in drop_dates], name='date').to_csv(DROP_CSV, index=False)
        with open(DROP_TXT, 'w') as f:
            for d in drop_dates:
                f.write(d.strftime('%Y-%m-%d') + '\n')
        print(f'Wrote drop dates: {DROP_CSV} ({len(drop_dates)} dates)')

        out = out[~out['date'].isin(drop_dates)].copy()
        out = out.sort_values(["station_id", "datetime"]).reset_index(drop=True)
        print(f'Filtered out rows for {len(drop_dates)} dates from output')

    # 6) 自检：每个 station 是否都变成完整网格长度
    expected_len = len(full_time)
    chk = out.groupby("station_id").size()
    bad = chk[chk != expected_len]
    if len(bad) > 0:
        print("WARNING: some stations not full length:")
        print(bad.head(20))
    else:
        print("OK: all stations full grid =", expected_len)

    # 缺失率统计（填充比例）
    miss_rate = out["miss_target"].mean()
    print("miss_target rate =", float(miss_rate))
    

    # 按站点+日期统计缺失情况：每个站-每天的总点数、原始缺失数、原始缺失率
    stats = out.groupby(["station", "date"]).agg(
        total_points=("miss_target", "size"),
        orig_missing=("miss_target", lambda x: int((x == 1).sum()))
    ).reset_index()
    stats["orig_missing_rate"] = stats["orig_missing"] / stats["total_points"]
    stats.to_csv('/root/autodl-tmp/datasets/missing_by_station_date.csv', index=False)
    print('saved: /root/autodl-tmp/datasets/missing_by_station_date.csv')

    # 另存按站点汇总（平均每日缺失率、总体缺失率）
    station_summary = stats.groupby("station").agg(
        days_reported=("date", "nunique"),
        avg_daily_missing_rate=("orig_missing_rate", "mean"),
        total_missing_points=("orig_missing", "sum")
    ).reset_index()
    station_summary.to_csv('/root/autodl-tmp/datasets/missing_by_station_summary.csv', index=False)
    print('saved: /root/autodl-tmp/datasets/missing_by_station_summary.csv')

    out.to_csv(OUT, index=False)
    print("saved:", OUT)
    print("saved:", MAP_OUT)

if __name__ == "__main__":
    main()
