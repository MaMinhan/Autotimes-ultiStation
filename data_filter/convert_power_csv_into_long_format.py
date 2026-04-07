#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python /root/autotimes/data_provider/convert_power_csv_into_long_format.py --input_dir /root/autodl-tmp/datasets/FY22 -
# -output_dir /root/autodl-tmp/datasets/FY22_long --station_from filename

import re
import argparse
from pathlib import Path
import pandas as pd


TIME_COL_RE = re.compile(r"^\d{2}:\d{2}$")


def guess_station_from_filename(fp: Path) -> str:
    """
    Example filename:
      Aberdeen 66_11kV FY2022.csv
      Balgowlah North 132_11kV FY2022.csv
    We'll strip the trailing ' FYxxxx' and extension.
    """
    name = fp.stem  # no .csv
    name = re.sub(r"\s+FY\d{4}$", "", name)  # remove trailing " FY2022"
    return name.strip()


def convert_one_file(input_csv: Path, output_csv: Path, station_from: str = "filename") -> int:
    df = pd.read_csv(input_csv)

    # Identify time columns like "00:15", "23:45", "00:00"
    time_cols = [c for c in df.columns if TIME_COL_RE.match(str(c))]
    if not time_cols:
        print(f"⚠️ Skip (no HH:MM columns): {input_csv}")
        return 0

    # Required columns
    if "Date" not in df.columns:
        raise ValueError(f"Column 'Date' not found in {input_csv}. Columns={list(df.columns)[:20]}...")

    # Station: prefer filename (your files are per substation), fallback to column if exists
    if station_from == "filename":
        station = guess_station_from_filename(input_csv)
    elif station_from == "column":
        if "Zone Substation" not in df.columns:
            raise ValueError(f"station_from=column but 'Zone Substation' not found in {input_csv}")
        # if file contains multiple stations (unlikely), keep per-row value
        station = None
    else:
        raise ValueError("station_from must be filename or column")

    # Parse Date (your sample is DD/MM/YYYY)
    df["Date"] = pd.to_datetime(
        df["Date"].astype(str).str.upper(),
        format="%d%b%Y",
        errors="coerce"
    )

    if df["Date"].isna().any():
        bad = df[df["Date"].isna()].head(3)
        raise ValueError(f"Some 'Date' values cannot be parsed in {input_csv}. Examples:\n{bad}")

    # Melt to long
    id_vars = ["Date"]
    keep_cols = []
    for c in ["Year", "Unit", "Zone Substation"]:
        if c in df.columns:
            keep_cols.append(c)
            id_vars.append(c)

    long_df = df.melt(
        id_vars=id_vars,
        value_vars=time_cols,
        var_name="time",
        value_name="target",
    )

    # Build datetime
    # time is "HH:MM"
    hm = long_df["time"].str.split(":", expand=True)
    long_df["hour"] = hm[0].astype(int)
    long_df["minute"] = hm[1].astype(int)
    long_df["datetime"] = long_df["Date"] + pd.to_timedelta(long_df["hour"], unit="h") + pd.to_timedelta(long_df["minute"], unit="m")

    # Station column
    if station is not None:
        long_df["station"] = station
    else:
        long_df["station"] = long_df["Zone Substation"].astype(str)

    # Keep only needed columns (baseline)
    out = long_df[["datetime", "station", "target"]].copy()
    out = out.sort_values(["station", "datetime"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    return len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder containing csv files (will scan recursively)")
    ap.add_argument("--output_dir", required=True, help="Where to write converted long csv files")
    ap.add_argument("--pattern", default="*.csv", help="File glob pattern, default *.csv")
    ap.add_argument("--station_from", choices=["filename", "column"], default="filename",
                    help="Station name source. Use filename for per-substation files.")
    ap.add_argument("--merge", action="store_true",
                    help="If set, also output a merged file 'power_long_all.csv' under output_dir")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.rglob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched under {input_dir} with pattern {args.pattern}")

    merged_path = output_dir / "power_long_all.csv"
    merged_written = False
    total_rows = 0

    for fp in files:
        # Preserve relative folder structure in output
        rel = fp.relative_to(input_dir)
        out_fp = (output_dir / rel).with_suffix("")  # remove .csv
        out_fp = out_fp.with_name(out_fp.name + "_long.csv")

        n = convert_one_file(fp, out_fp, station_from=args.station_from)
        total_rows += n
        print(f"✅ {fp} -> {out_fp}  ({n} rows)")

        if args.merge:
            # Append to merged CSV (streaming append)
            mode = "a" if merged_written else "w"
            header = not merged_written
            df_chunk = pd.read_csv(out_fp)  # ok for per-file; if huge, do chunksize
            df_chunk.to_csv(merged_path, index=False, mode=mode, header=header)
            merged_written = True

    print(f"\nDone. Converted files: {len(files)}, total rows: {total_rows}")
    if args.merge:
        print(f"Merged file: {merged_path}")


if __name__ == "__main__":
    main()
