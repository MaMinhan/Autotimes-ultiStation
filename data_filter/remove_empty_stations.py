#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--infile', default='/root/autodl-tmp/datasets/merged_FY22_24_long_filtered.csv')
    p.add_argument('--out', default='/root/autodl-tmp/datasets/merged_FY22_24_long_filtered_noempty.csv')
    p.add_argument('--dropped', default='/root/autodl-tmp/datasets/dropped_all_empty_stations.csv')
    args = p.parse_args()

    infile = Path(args.infile)
    if not infile.exists():
        print('Input file not found:', infile)
        return

    df = pd.read_csv(infile, parse_dates=['datetime'])
    if 'station' not in df.columns:
        print('Input does not contain `station` column; cannot group.')
        return
    if 'target' not in df.columns:
        print('Input does not contain `target` column; nothing to check.')
        return

    # Count non-missing targets per station
    non_missing = df.groupby('station')['target'].apply(lambda x: x.notna().sum())
    empty_stations = non_missing[non_missing == 0].index.tolist()

    if not empty_stations:
        print('No fully-empty stations found.')
        # still save a copy
        df.to_csv(args.out, index=False)
        print('Saved copy to', args.out)
        return

    print(f'Found {len(empty_stations)} fully-empty stations; removing them:')
    for s in empty_stations:
        print('  -', s)

    # Filter
    df_filtered = df[~df['station'].isin(empty_stations)].copy()

    # Save outputs
    df_filtered.to_csv(args.out, index=False)
    pd.Series(empty_stations, name='station').to_csv(args.dropped, index=False)

    print('Saved filtered file to', args.out)
    print('Saved dropped stations list to', args.dropped)

if __name__ == '__main__':
    main()
