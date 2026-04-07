#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser(description='Add station_id to merged CSV')
    p.add_argument('--infile', default='/root/autodl-tmp/datasets/merged_clean.csv')
    p.add_argument('--out', default='/root/autodl-tmp/datasets/merged_clean_with_id.csv')
    p.add_argument('--map_out', default='/root/autodl-tmp/datasets/station_map.csv')
    args = p.parse_args()

    infile = Path(args.infile)
    out = Path(args.out)
    map_out = Path(args.map_out)

    if not infile.exists():
        print('Input file not found:', infile)
        return

    df = pd.read_csv(infile, parse_dates=['datetime'], low_memory=False)
    if 'station' not in df.columns:
        print('No `station` column in', infile)
        return

    stations = sorted(df['station'].astype(str).unique().tolist())
    station2id = {s: i for i, s in enumerate(stations)}

    df['station_id'] = df['station'].map(station2id).astype(int)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print('Saved file with station_id to', out)

    map_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'station': stations, 'station_id': list(range(len(stations)))})\
        .to_csv(map_out, index=False)
    print('Saved station map to', map_out)

    print('\nSummary:')
    print('Total stations:', len(stations))
    print('\nSample mapping (first 20):')
    for s, i in list(station2id.items())[:20]:
        print(f'  {i}: {s}')

if __name__ == '__main__':
    main()
