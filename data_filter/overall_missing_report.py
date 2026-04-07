#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--infile', default='/root/autodl-tmp/datasets/merged_FY22_24_long_filtered.csv')
    p.add_argument('--out_dir', default='/root/autodl-tmp/datasets')
    args = p.parse_args()

    infile = Path(args.infile)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not infile.exists():
        print('Input file not found:', infile)
        return

    df = pd.read_csv(infile, parse_dates=['datetime'], low_memory=False)
    if 'target' not in df.columns:
        print('Input missing `target` column')
        return

    total = len(df)
    missing = int(df['target'].isna().sum())
    miss_rate = missing / total if total > 0 else 0.0

    print('Overall points:', total)
    print('Overall missing points:', missing)
    print('Overall missing rate: {:.4f} ({:.2%})'.format(miss_rate, miss_rate))

    # per-station summary
    if 'station' in df.columns:
        ssum = df.groupby('station')['target'].agg(total_points='size', missing_points=lambda x: int(x.isna().sum())).reset_index()
        ssum['missing_rate'] = ssum['missing_points'] / ssum['total_points']
        sfile = out_dir / 'missing_by_station_overall.csv'
        ssum.to_csv(sfile, index=False)
        print('Saved per-station summary to', sfile)
        print('\nTop 10 stations by missing_rate:')
        print(ssum.sort_values('missing_rate', ascending=False).head(10).to_string(index=False))
    else:
        print('No `station` column; skipping per-station summary')

    # per-date summary
    if 'datetime' in df.columns:
        df['date'] = df['datetime'].dt.floor('D')
        dsum = df.groupby('date')['target'].agg(total_points='size', missing_points=lambda x: int(x.isna().sum())).reset_index()
        dsum['missing_rate'] = dsum['missing_points'] / dsum['total_points']
        dfile = out_dir / 'missing_by_date_overall.csv'
        dsum.to_csv(dfile, index=False)
        print('Saved per-date summary to', dfile)
        print('\nTop 10 dates by missing_rate:')
        print(dsum.sort_values('missing_rate', ascending=False).head(10).to_string(index=False))
    else:
        print('No `datetime` column; skipping per-date summary')

if __name__ == '__main__':
    main()
