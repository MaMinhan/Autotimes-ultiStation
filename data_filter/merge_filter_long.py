#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import re
import os

TIME_COL_RE = re.compile(r"^\d{1,2}:\d{2}$")

def infer_station_from_filename(fp: Path):
    name = fp.stem
    # remove trailing _long or FY suffixes
    name = re.sub(r"(_long$)|(_long\.csv$)", "", name)
    name = re.sub(r"\s+FY\d{4}$", "", name)
    return name.strip()

def read_long_file(fp: Path):
    # Expect long format with datetime,station,target; fall back to wide->long if needed
    try:
        df = pd.read_csv(fp, parse_dates=['datetime'])
    except Exception:
        df = pd.read_csv(fp)

    if 'datetime' in df.columns and 'target' in df.columns:
        return df

    # try wide format
    time_cols = [c for c in df.columns if TIME_COL_RE.match(str(c))]
    if 'Date' in df.columns and time_cols:
        long = df.melt(id_vars=['Date'], value_vars=time_cols, var_name='time', value_name='target')
        long['Date'] = pd.to_datetime(long['Date'], errors='coerce')
        tm = long['time'].str.split(':', expand=True)
        long['hour'] = tm[0].astype(int)
        long['minute'] = tm[1].astype(int)
        long['datetime'] = long['Date'] + pd.to_timedelta(long['hour'], unit='h') + pd.to_timedelta(long['minute'], unit='m')
        return long

    raise ValueError(f'unrecognized file format: {fp}')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dirs', nargs='+', default=[
        '/root/autodl-tmp/datasets/FY22_long',
        '/root/autodl-tmp/datasets/FY23_long',
        '/root/autodl-tmp/datasets/FY24_long',
    ])
    p.add_argument('--station_summary', default='/root/autodl-tmp/datasets/missing_station.csv')
    p.add_argument('--date_summary', default='/root/autodl-tmp/datasets/missing_dates.csv')
    p.add_argument('--station_thresh', type=float, default=0.10)
    p.add_argument('--date_thresh', type=float, default=0.30)
    p.add_argument('--out', default='/root/autodl-tmp/datasets/merged_FY22_24_long_filtered.csv')
    args = p.parse_args()

    # load station summary
    drop_stations = set()
    if os.path.exists(args.station_summary):
        try:
            ss = pd.read_csv(args.station_summary)
            if 'avg_daily_missing_rate' in ss.columns and 'station' in ss.columns:
                # Stations with avg_daily_missing_rate == 1.0 -> drop all their data
                full_missing = ss[ss['avg_daily_missing_rate'] >= 0.999999]['station'].astype(str).tolist()
                if full_missing:
                    drop_stations.update(full_missing)
                    print(f'Will drop {len(full_missing)} stations with avg_daily_missing_rate == 1.0:')
                    for s in full_missing:
                        print('  DROP (all data):', s)

                # Stations exceeding threshold
                bad = ss[ss['avg_daily_missing_rate'] > args.station_thresh]
                bad_list = bad['station'].astype(str).tolist()
                if bad_list:
                    drop_stations.update(bad_list)
                    print(f'Will drop {len(bad_list)} stations with avg_daily_missing_rate > {args.station_thresh}')
        except Exception as e:
            print('Failed reading station_summary:', e)
    else:
        print('station_summary not found, no station-level filtering')

    # load date summary
    drop_dates = set()
    if os.path.exists(args.date_summary):
        try:
            ds = pd.read_csv(args.date_summary, parse_dates=['date'])
            if 'pct_stations_missing' in ds.columns:
                badd = ds[ds['pct_stations_missing'] > args.date_thresh]
                drop_dates = set(pd.to_datetime(badd['date']).dt.date.tolist())
                print(f'Will drop {len(drop_dates)} dates with pct_stations_missing > {args.date_thresh}')
        except Exception as e:
            print('Failed reading date_summary:', e)
    else:
        print('date_summary not found, no date-level filtering')

    parts = []
    dropped_stations = []
    dropped_dates_global = set()

    for d in args.dirs:
        dpath = Path(d)
        if not dpath.exists():
            print('skip missing dir', dpath)
            continue
        for fp in sorted(dpath.glob('*_long.csv')):
            try:
                df = read_long_file(fp)
            except Exception as e:
                print('skip file', fp, 'error:', e)
                continue

            # ensure datetime
            if 'datetime' not in df.columns:
                print('skip file without datetime:', fp)
                continue

            # station name
            if 'station' in df.columns:
                station = str(df['station'].iloc[0]) if not df['station'].isna().all() else infer_station_from_filename(fp)
            else:
                station = infer_station_from_filename(fp)

            if station in drop_stations:
                print(f'Skipping whole station {station} (from file {fp.name}) due to high missing rate')
                dropped_stations.append(station)
                continue

            # drop rows with dates in drop_dates
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.dropna(subset=['datetime']).copy()
            if len(drop_dates) > 0:
                dates_in_file = set(df['datetime'].dt.date.unique())
                intersect = sorted(list(dates_in_file & drop_dates))
                if intersect:
                    before = len(df)
                    df = df[~df['datetime'].dt.date.isin(intersect)].copy()
                    after = len(df)
                    dropped_dates_global.update(intersect)
                    print(f'From file {fp.name} (station={station}) dropped dates {intersect}, removed {before-after} rows')

            # If after filtering there are no rows, or all target values are NaN, skip this station
            if df.empty:
                print(f'Skipping station {station} (from file {fp.name}) because no rows remain after date filtering')
                dropped_stations.append(station)
                continue
            if 'target' in df.columns and df['target'].notna().sum() == 0:
                print(f'Skipping station {station} (from file {fp.name}) because all target values are missing after filtering')
                dropped_stations.append(station)
                continue

            # ensure station col
            if 'station' not in df.columns:
                df['station'] = station

            # keep only necessary columns
            keep_cols = [c for c in ['datetime','station','target'] if c in df.columns]
            df = df[keep_cols].copy()
            parts.append(df)

    if not parts:
        print('No data to merge after filtering')
        return

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(['station','datetime']).reset_index(drop=True)

    out.to_csv(args.out, index=False)
    print('Saved merged filtered file:', args.out)

    if dropped_stations:
        dsfile = Path(args.out).parent / 'dropped_stations.csv'
        pd.Series(sorted(set(dropped_stations)), name='station').to_csv(dsfile, index=False)
        print('Wrote dropped stations to', dsfile)

    if dropped_dates_global:
        ddfile = Path(args.out).parent / 'dropped_dates.csv'
        pd.Series(sorted([d.strftime('%Y-%m-%d') for d in dropped_dates_global]), name='date').to_csv(ddfile, index=False)
        print('Wrote dropped dates to', ddfile)

if __name__ == '__main__':
    main()
