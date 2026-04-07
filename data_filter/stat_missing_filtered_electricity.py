#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser(description='Compute missingness for a filtered electricity CSV')
    p.add_argument('--infile', default='/root/autodl-tmp/datasets/SelfMadeAusgridData/FY22_24_filtered_electricity.csv')
    p.add_argument('--out_dir', default='/root/autodl-tmp/datasets')
    p.add_argument('--save', action='store_true', help='Save per-station and per-date summaries')
    args = p.parse_args()

    infile = Path(args.infile)
    out_dir = Path(args.out_dir)
    if not infile.exists():
        print('Input not found:', infile)
        return

    df = pd.read_csv(infile, parse_dates=['datetime'], low_memory=False)
    if 'target' not in df.columns:
        print('Input missing `target` column')
        return

    total = len(df)
    missing = int(df['target'].isna().sum())
    miss_rate = missing / total if total > 0 else 0.0

    print('File:', infile)
    print('Total rows:', total)
    print('Missing target rows:', missing)
    print('Overall missing rate: {:.4f} ({:.2%})'.format(miss_rate, miss_rate))

    if 'station' in df.columns:
        ssum = df.groupby('station')['target'].agg(total_points='size', missing_points=lambda x: int(x.isna().sum())).reset_index()
        ssum['missing_rate'] = ssum['missing_points'] / ssum['total_points']
        top_stations = ssum.sort_values('missing_rate', ascending=False).head(20)
        print('\nTop 20 stations by missing_rate:')
        print(top_stations.to_string(index=False))
        if args.save:
            out_dir.mkdir(parents=True, exist_ok=True)
            sfile = out_dir / 'missing_by_station_filtered_electricity.csv'
            ssum.to_csv(sfile, index=False)
            print('Saved per-station summary to', sfile)
    else:
        print('No `station` column found; skipping per-station summary')

    if 'datetime' in df.columns:
        df['date'] = df['datetime'].dt.floor('D')
        dsum = df.groupby('date')['target'].agg(total_points='size', missing_points=lambda x: int(x.isna().sum())).reset_index()
        dsum['missing_rate'] = dsum['missing_points'] / dsum['total_points']
        top_dates = dsum.sort_values('missing_rate', ascending=False).head(20)
        print('\nTop 20 dates by missing_rate:')
        print(top_dates.to_string(index=False))
        if args.save:
            out_dir.mkdir(parents=True, exist_ok=True)
            dfile = out_dir / 'missing_by_date_filtered_electricity.csv'
            dsum.to_csv(dfile, index=False)
            print('Saved per-date summary to', dfile)
    else:
        print('No `datetime` column found; skipping per-date summary')

if __name__ == '__main__':
    main()
