#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser(description='Drop stations whose `target` is all NaN and save filtered CSV')
    p.add_argument('--infile', default='/root/autodl-tmp/datasets/merged_FY22_24_long_filtered.csv')
    p.add_argument('--out', default='/root/autodl-tmp/datasets/merged_FY22_24_long_filtered_drop_allnan.csv')
    p.add_argument('--dropped', default='/root/autodl-tmp/datasets/dropped_allnan_stations.csv')
    args = p.parse_args()

    infile = Path(args.infile)
    if not infile.exists():
        print('Input file not found:', infile)
        return

    df = pd.read_csv(infile, parse_dates=['datetime'], low_memory=False)
    if 'station' not in df.columns or 'target' not in df.columns:
        print('Input must contain `station` and `target` columns')
        return

    grp = df.groupby('station')['target'].apply(lambda x: x.isna().all())
    allnan_stations = grp[grp].index.tolist()

    if not allnan_stations:
        print('No stations found with all-NaN target.')
        # still save a copy
        df.to_csv(args.out, index=False)
        print('Saved copy to', args.out)
        return

    print(f'Found {len(allnan_stations)} stations with all-NaN target:')
    for s in allnan_stations:
        print('  -', s)

    # save dropped list
    Path(args.dropped).parent.mkdir(parents=True, exist_ok=True)
    pd.Series(allnan_stations, name='station').to_csv(args.dropped, index=False)

    # filter df
    df_filtered = df[~df['station'].isin(allnan_stations)].copy()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(args.out, index=False)

    print('Saved filtered CSV to', args.out)
    print('Saved dropped stations list to', args.dropped)

if __name__ == '__main__':
    main()
