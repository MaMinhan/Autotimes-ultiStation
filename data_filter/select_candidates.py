#!/usr/bin/env python3
import argparse
import pandas as pd

IN = '/root/autodl-tmp/datasets/missing_by_station_summary.csv'
OUT = '/root/autodl-tmp/datasets/candidate_stations.csv'

def main():
    p = argparse.ArgumentParser(description='Select candidate stations by avg daily missing rate')
    p.add_argument('--threshold', type=float, default=0.30, help='max avg_daily_missing_rate allowed (default 0.05)')
    p.add_argument('--top', type=int, default=20, help='print top N candidates')
    args = p.parse_args()

    df = pd.read_csv(IN)
    if 'avg_daily_missing_rate' not in df.columns:
        raise SystemExit('missing column avg_daily_missing_rate in ' + IN)

    cand = df[df['avg_daily_missing_rate'] <= args.threshold].copy()
    cand = cand.sort_values('avg_daily_missing_rate')
    cand.to_csv(OUT, index=False)

    print(f'saved: {OUT}')
    print(f'found {len(cand)} candidate stations (threshold={args.threshold})')
    if len(cand) > 0:
        print('\nTop candidates:')
        print(cand.head(args.top).to_string(index=False))

if __name__ == '__main__':
    main()
