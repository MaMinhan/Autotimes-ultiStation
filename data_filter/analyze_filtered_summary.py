#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='infile', default='/root/autodl-tmp/datasets/SelfMadeAusgridData/FY22_24_filtered_electricity.csv')
    p.add_argument('--out_dir', default='/root/autodl-tmp/datasets')
    args = p.parse_args()

    infile = Path(args.infile)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not infile.exists():
        print('Input file not found:', infile)
        return

    df = pd.read_csv(infile, parse_dates=['datetime'])
    if 'target' not in df.columns:
        print('Input missing `target` column')
        return

    df['date'] = df['datetime'].dt.floor('D')

    # per-station-per-date stats
    stats = df.groupby(['station','date']).agg(
        total_points=('target','size'),
        orig_missing=('target', lambda x: int(x.isna().sum()))
    ).reset_index()
    stats['orig_missing_rate'] = stats['orig_missing'] / stats['total_points']
    sfile = out_dir / 'missing_by_station_date_filtered.csv'
    stats.to_csv(sfile, index=False)
    print('Saved per-station-per-date stats to', sfile)

    # per-date summary
    per_date = stats.groupby('date').agg(
        stations_with_missing=('orig_missing', lambda x: int((x > 0).sum())),
        total_missing_points=('orig_missing', 'sum'),
        stations_reported=('station', 'nunique')
    ).reset_index()
    per_date['pct_stations_missing'] = per_date['stations_with_missing'] / per_date['stations_reported']
    dfile = out_dir / 'missing_by_date_summary_filtered.csv'
    per_date.to_csv(dfile, index=False)
    print('Saved per-date summary to', dfile)

    # per-station summary
    station_summary = stats.groupby('station').agg(
        days_reported=('date','nunique'),
        avg_daily_missing_rate=('orig_missing_rate','mean'),
        total_missing_points=('orig_missing','sum')
    ).reset_index()
    ssum = out_dir / 'missing_by_station_summary_filtered.csv'
    station_summary.to_csv(ssum, index=False)
    print('Saved per-station summary to', ssum)

    # print top issues
    print('\nTop 10 stations by avg_daily_missing_rate:')
    print(station_summary.sort_values('avg_daily_missing_rate', ascending=False).head(10).to_string(index=False))

    print('\nTop 10 dates by stations_with_missing:')
    print(per_date.sort_values('stations_with_missing', ascending=False).head(10).to_string(index=False))

if __name__ == '__main__':
    main()
