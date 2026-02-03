#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser(description='Remove dates where all stations have missing data')
    p.add_argument('--infile', default='/root/autodl-tmp/datasets/12点35/merged_12点35_drop13allnan_station.csv')
    p.add_argument('--out', default='/root/autodl-tmp/datasets/12点35/merged_12点35_drop13allnan_station_and_nandate.csv')
    p.add_argument('--dropped', default='/root/autodl-tmp/datasets/12点35/dropped_all_empty_dates.csv')
    args = p.parse_args()

    infile = Path(args.infile)
    if not infile.exists():
        print('Input file not found:', infile)
        return

    df = pd.read_csv(infile, parse_dates=['datetime'], low_memory=False)
    if 'station' not in df.columns or 'target' not in df.columns:
        print('Input must contain `station` and `target` columns')
        return

    df['date'] = df['datetime'].dt.floor('D')

    stations = sorted(df['station'].astype(str).unique().tolist())
    total_stations = len(stations)
    print(f'Total stations in file: {total_stations}')

    # compute non-missing counts per station-date
    grp = df.groupby(['station','date'])['target'].apply(lambda x: int(x.notna().sum())).reset_index(name='non_missing')

    # build mapping for quick lookup
    mapping = {(row['station'], row['date'].date() if hasattr(row['date'], 'date') else pd.to_datetime(row['date']).date()): row['non_missing'] for _, row in grp.iterrows()}

    unique_dates = sorted(df['date'].dt.date.unique())
    drop_dates = []
    missing_summary = []

    for d in unique_dates:
        stations_missing = 0
        for s in stations:
            nm = mapping.get((s, d), 0)
            if nm == 0:
                stations_missing += 1
        missing_summary.append({'date': d.strftime('%Y-%m-%d'), 'stations_with_missing': stations_missing, 'stations_total': total_stations, 'pct_stations_missing': stations_missing/total_stations})
        if stations_missing == total_stations:
            drop_dates.append(d)

    # save per-date missing summary
    summary_df = pd.DataFrame(missing_summary)
    summary_out = Path(args.dropped).parent / 'missing_by_date_allcheck.csv'
    summary_df.to_csv(summary_out, index=False)
    print('Saved per-date missing summary to', summary_out)

    if not drop_dates:
        print('No fully-empty dates found (no date where all stations missing).')
        # still copy input to out
        df.drop(columns=['date']).to_csv(args.out, index=False)
        print('Saved copy to', args.out)
        return

    print(f'Found {len(drop_dates)} fully-empty dates to drop:')
    for d in drop_dates:
        print('  -', d.strftime('%Y-%m-%d'))

    # write dropped dates file
    Path(args.dropped).parent.mkdir(parents=True, exist_ok=True)
    pd.Series([d.strftime('%Y-%m-%d') for d in drop_dates], name='date').to_csv(args.dropped, index=False)

    # filter df
    df_filtered = df[~df['date'].dt.date.isin(drop_dates)].copy()
    df_filtered = df_filtered.drop(columns=['date'])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(args.out, index=False)
    print('Saved filtered CSV to', args.out)

if __name__ == '__main__':
    main()
