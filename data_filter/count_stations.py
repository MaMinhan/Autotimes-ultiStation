#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser(description='Count unique stations in a CSV with columns including `station`')
    p.add_argument('--infile', default='/root/autodl-tmp/datasets/SelfMadeAusgridData/FY22_24_filtered_electricity.csv')
    p.add_argument('--save_list', action='store_true', help='Save station list to <infile>_stations.csv')
    args = p.parse_args()

    infile = Path(args.infile)
    if not infile.exists():
        print('Input not found:', infile)
        return

    df = pd.read_csv(infile, usecols=['station'] if 'station' in pd.read_csv(infile, nrows=0).columns else None)
    if 'station' not in df.columns:
        print('No `station` column found in', infile)
        return

    stations = df['station'].astype(str).unique()
    print('Unique stations count:', len(stations))
    if args.save_list:
        out = infile.parent / (infile.stem + '_stations.csv')
        pd.Series(sorted(stations), name='station').to_csv(out, index=False)
        print('Saved station list to', out)

if __name__ == '__main__':
    main()
