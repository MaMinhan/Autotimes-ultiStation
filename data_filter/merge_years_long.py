#!/usr/bin/env python3
"""
Merge all *_long.csv files from given FY folders into one long CSV.

Usage:
  python merge_years_long.py --dirs /root/autodl-tmp/datasets/FY22_long /root/autodl-tmp/datasets/FY23_long /root/autodl-tmp/datasets/FY24_long

The script supports files already in long format (columns include `datetime` and `target`)
or wide format (a `Date` column and time columns like `00:15`). It will try to parse both.
"""
import argparse
from pathlib import Path
import pandas as pd
import re

TIME_COL_RE = re.compile(r"^\d{1,2}:\d{2}$")

def infer_station_from_filename(fp: Path):
    name = fp.stem
    name = re.sub(r"\s+FY\d{4}$", "", name)
    name = re.sub(r"(_long$)|(_long\.csv$)", "", name)
    return name.strip()

def to_long(fp: Path):
    df = pd.read_csv(fp)
    # long format
    if 'datetime' in df.columns and 'target' in df.columns:
        out = df[['datetime', 'target']].copy()
        # if station column exists, keep it; otherwise infer later
        if 'station' in df.columns:
            out['station'] = df['station']
        out['datetime'] = pd.to_datetime(out['datetime'], errors='coerce')
        return out

    # wide format: Date + time columns like 00:15
    time_cols = [c for c in df.columns if TIME_COL_RE.match(str(c))]
    if 'Date' in df.columns and time_cols:
        long = df.melt(id_vars=['Date'], value_vars=time_cols, var_name='time', value_name='target')
        long['Date'] = pd.to_datetime(long['Date'], errors='coerce')
        hm = long['time'].str.split(':', expand=True)
        long['hour'] = hm[0].astype(int)
        long['minute'] = hm[1].astype(int)
        long['datetime'] = long['Date'] + pd.to_timedelta(long['hour'], unit='h') + pd.to_timedelta(long['minute'], unit='m')
        return long[['datetime', 'target']]

    raise ValueError(f'Unrecognized file format: {fp}')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dirs', nargs='+', required=False, default=[
        '/root/autodl-tmp/datasets/FY22_long',
        '/root/autodl-tmp/datasets/FY23_long',
        '/root/autodl-tmp/datasets/FY24_long',
    ])
    p.add_argument('--pattern', default='*_long.csv')
    p.add_argument('--out', default='/root/autodl-tmp/datasets/merged_FY22_24_long.csv')
    args = p.parse_args()

    parts = []
    for d in args.dirs:
        dpath = Path(d)
        if not dpath.exists():
            print('skip missing dir', dpath)
            continue
        files = sorted(dpath.glob(args.pattern))
        if not files:
            print('no matching files in', dpath)
            continue

        for fp in files:
            try:
                long = to_long(fp)
            except Exception as e:
                print('skip', fp.name, 'error:', e)
                continue

            # ensure datetime parsed
            long = long.dropna(subset=['datetime']).copy()
            if long.empty:
                print('skip', fp.name, '— no valid datetimes')
                continue

            # station
            if 'station' not in long.columns:
                station = infer_station_from_filename(fp)
                long['station'] = station

            # keep only needed cols
            long = long[['datetime', 'station', 'target']]
            parts.append(long)
            print('added', fp.name, 'rows=', len(long))

    if not parts:
        print('no data merged')
        return

    out = pd.concat(parts, ignore_index=True)
    out['datetime'] = pd.to_datetime(out['datetime'], errors='coerce')
    out = out.dropna(subset=['datetime']).sort_values(['station','datetime']).reset_index(drop=True)

    out.to_csv(args.out, index=False)
    print('saved merged file to', args.out, 'rows=', len(out))

if __name__ == '__main__':
    main()
