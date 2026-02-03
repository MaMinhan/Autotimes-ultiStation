#!/usr/bin/env python3
import pandas as pd

IN = '/root/autodl-tmp/datasets/missing_by_station_date.csv'
OUT = '/root/autodl-tmp/datasets/missing_by_date_summary.csv'

def main():
    df = pd.read_csv(IN, parse_dates=['date'])

    # stations with any missing that day
    per_date = df.groupby('date').agg(
        stations_with_missing = ('orig_missing', lambda x: int((x > 0).sum())),
        total_missing_points = ('orig_missing', 'sum'),
        stations_reported = ('station', 'nunique')
    ).reset_index()

    per_date['pct_stations_missing'] = per_date['stations_with_missing'] / per_date['stations_reported']

    per_date = per_date.sort_values(['stations_with_missing', 'total_missing_points'], ascending=False)
    per_date.to_csv(OUT, index=False)

    print('saved:', OUT)
    print('\nTop 10 dates by number of stations missing:')
    print(per_date.head(10)[['date','stations_with_missing','stations_reported','pct_stations_missing']].to_string(index=False))

    print('\nTop 10 dates by total missing points:')
    print(per_date.sort_values('total_missing_points', ascending=False).head(10)[['date','total_missing_points','stations_with_missing']].to_string(index=False))

if __name__ == '__main__':
    main()
