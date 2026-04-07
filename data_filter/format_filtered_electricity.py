#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

TARGET_CANDIDATES = ['target', 'load', 'value', 'power', 'energy']

def find_target_col(df: pd.DataFrame):
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    # fallback: pick first numeric column that's not datetime
    for c in df.columns:
        if c.lower() in ('datetime','date','time','station'):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def main():
    p = argparse.ArgumentParser(description='Format filtered electricity CSV to datetime|station|station_id|target|miss_target')
    p.add_argument('--infile', default='/root/autodl-tmp/datasets/SelfMadeAusgridData/FY22_24_filtered_electricity.csv')
    p.add_argument('--out', default='/root/autodl-tmp/datasets/merged_clean.csv')
    p.add_argument('--map_out', default='/root/autodl-tmp/datasets/station_map.csv')
    args = p.parse_args()

    infile = Path(args.infile)
    out = Path(args.out)
    map_out = Path(args.map_out)

    if not infile.exists():
        print('Input not found:', infile)
        return

    df = pd.read_csv(infile, low_memory=False)

    # parse datetime
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    elif 'date' in df.columns and 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
    elif 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        # try to infer a datetime-like column
        for c in df.columns:
            if 'date' in c.lower() or 'time' in c.lower():
                df['datetime'] = pd.to_datetime(df[c], errors='coerce')
                break

    if 'station' not in df.columns:
        # try to infer station column name
        possible = [c for c in df.columns if 'station' in c.lower()]
        if possible:
            df['station'] = df[possible[0]].astype(str)
        else:
            print('No station column found in', infile)
            return

    target_col = find_target_col(df)
    if target_col is None:
        print('No target-like column found in', infile)
        return

    # compute miss_target before any modification
    df['miss_target'] = df[target_col].isna().astype(int)

    # rename target column
    df = df.rename(columns={target_col: 'target'})

    # ensure datetime present
    if 'datetime' not in df.columns:
        print('datetime column could not be parsed/found')
        return

    # build station_id mapping (stable sorted)
    stations = sorted(df['station'].astype(str).unique().tolist())
    station2id = {s: i for i, s in enumerate(stations)}
    df['station_id'] = df['station'].map(station2id).astype(int)

    # select and reorder columns
    out_df = df[['datetime', 'station', 'station_id', 'target', 'miss_target']].copy()

    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print('Saved formatted CSV to', out, 'rows=', len(out_df))

    map_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'station': stations, 'station_id': list(range(len(stations)))})\
        .to_csv(map_out, index=False)
    print('Saved station map to', map_out, 'stations=', len(stations))

if __name__ == '__main__':
    main()
