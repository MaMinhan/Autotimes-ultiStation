#!/usr/bin/env python3
"""
Traverse given year folders (wide or long CSVs) and compute missingness
by station and by date. Outputs:
 - /root/autodl-tmp/datasets/missing_by_station_date_years.csv
 - /root/autodl-tmp/datasets/missing_by_date_summary_years.csv
 - /root/autodl-tmp/datasets/missing_by_station_summary_years.csv

Usage:
  python stat_missing_years.py --dirs /root/autodl-tmp/datasets/FY22 /root/autodl-tmp/datasets/FY23 /root/autodl-tmp/datasets/FY24

The script attempts to handle two formats:
 - long format: columns include `datetime` and `target` (and optionally `station`)
 - wide format: a `Date` column and many time columns like `00:15`, `00:30` etc.

If station name is not present in the file it will be inferred from filename.
"""
import argparse
import pandas as pd
import numpy as np
import re
from pathlib import Path

TIME_COL_RE = re.compile(r"^\d{1,2}:\d{2}$")

def infer_station_from_filename(fp: Path):
    name = fp.stem
    name = re.sub(r"\s+FY\d{4}$", "", name)
    return name.strip()

def read_to_long(fp: Path):
    """Read a CSV file and return DataFrame with columns `datetime`, `target`, `station`"""
    df = pd.read_csv(fp)
    # If long-format
    if 'datetime' in df.columns and 'target' in df.columns:
        out = df[['datetime', 'target']].copy()
        out['datetime'] = pd.to_datetime(out['datetime'], errors='coerce')
        return out

    # wide-format (Date + time columns like 00:15)
    time_cols = [c for c in df.columns if TIME_COL_RE.match(str(c))]
    if 'Date' in df.columns and time_cols:
        id_vars = ['Date']
        long = df.melt(id_vars=id_vars, value_vars=time_cols, var_name='time', value_name='target')
        # parse Date (try common formats)
        long['Date'] = pd.to_datetime(long['Date'], errors='coerce')
        tm = long['time'].str.split(':', expand=True)
        long['hour'] = tm[0].astype(int)
        long['minute'] = tm[1].astype(int)
        long['datetime'] = long['Date'] + pd.to_timedelta(long['hour'], unit='h') + pd.to_timedelta(long['minute'], unit='m')
        return long[['datetime', 'target']].copy()

    # else: try columns like 'date' lower-case
    if 'date' in df.columns and time_cols:
        id_vars = ['date']
        long = df.melt(id_vars=id_vars, value_vars=time_cols, var_name='time', value_name='target')
        long['date'] = pd.to_datetime(long['date'], errors='coerce')
        tm = long['time'].str.split(':', expand=True)
        long['hour'] = tm[0].astype(int)
        long['minute'] = tm[1].astype(int)
        long['datetime'] = long['date'] + pd.to_timedelta(long['hour'], unit='h') + pd.to_timedelta(long['minute'], unit='m')
        return long[['datetime', 'target']].copy()

    raise ValueError(f'Unrecognized file format: {fp}')

def process_file(fp: Path):
    try:
        long = read_to_long(fp)
    except Exception as e:
        print(f'Skipping {fp.name}: {e}')
        return None

    station = None
    # if file contains station column, use it
    # but our read_to_long dropped other columns; infer from filename
    station = infer_station_from_filename(fp)

    long['target'] = pd.to_numeric(long['target'], errors='coerce')
    long = long.dropna(subset=['datetime'])
    if len(long) == 0:
        return None

    long['date'] = long['datetime'].dt.floor('D')
    grp = long.groupby('date').agg(
        total_points=('target', 'size'),
        orig_missing=('target', lambda x: int(x.isna().sum()))
    ).reset_index()
    grp['station'] = station
    grp['orig_missing_rate'] = grp['orig_missing'] / grp['total_points']
    # reorder
    grp = grp[['station', 'date', 'total_points', 'orig_missing', 'orig_missing_rate']]
    return grp

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dirs', nargs='+', required=True, help='Folders to scan (FY22 FY23 FY24)')
    p.add_argument('--out_dir', default='/root/autodl-tmp/datasets', help='Where to write outputs')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    for d in args.dirs:
        dpath = Path(d)
        if not dpath.exists():
            print('Skip missing folder', dpath)
            continue
        for fp in sorted(dpath.glob('*.csv')):
            print('Processing', fp)
            grp = process_file(fp)
            if grp is not None:
                parts.append(grp)

    if not parts:
        print('No data processed.')
        return

    stats = pd.concat(parts, ignore_index=True)
    stats = stats.sort_values(['station', 'date']).reset_index(drop=True)

    # save per-station-per-date
    sfile = out_dir / 'missing_by_station_date_years.csv'
    stats.to_csv(sfile, index=False)
    print('saved:', sfile)

    # per-date summary
    per_date = stats.groupby('date').agg(
        stations_with_missing=('orig_missing', lambda x: int((x > 0).sum())),
        total_missing_points=('orig_missing', 'sum'),
        stations_reported=('station', 'nunique')
    ).reset_index()
    per_date['pct_stations_missing'] = per_date['stations_with_missing'] / per_date['stations_reported']
    dfile = out_dir / 'missing_by_date_summary_years.csv'
    per_date.to_csv(dfile, index=False)
    print('saved:', dfile)

    # per-station summary
    station_summary = stats.groupby('station').agg(
        days_reported=('date', 'nunique'),
        avg_daily_missing_rate=('orig_missing_rate', 'mean'),
        total_missing_points=('orig_missing', 'sum')
    ).reset_index()
    ssum = out_dir / 'missing_by_station_summary_years.csv'
    station_summary.to_csv(ssum, index=False)
    print('saved:', ssum)

    # print top problematic dates
    print('\nTop 10 dates by stations_with_missing:')
    print(per_date.sort_values('stations_with_missing', ascending=False).head(10).to_string(index=False))

    print('\nTop 10 dates by total_missing_points:')
    print(per_date.sort_values('total_missing_points', ascending=False).head(10).to_string(index=False))

if __name__ == '__main__':
    main()
