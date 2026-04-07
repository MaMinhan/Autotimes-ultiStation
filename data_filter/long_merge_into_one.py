import pandas as pd
from pathlib import Path

# Define input directories for FY22, FY23, FY24
dirs = [
    "/root/autodl-tmp/datasets/FY22_long", 
    "/root/autodl-tmp/datasets/FY23_long", 
    "/root/autodl-tmp/datasets/FY24_long"
]

# Output path for the merged file
out_path = Path("/root/autodl-tmp/datasets/merged_FY22_24_long.csv")

# Initialize a list to hold all dataframes
dfs = []

# Loop through each directory (FY22, FY23, FY24) and collect the CSV files
for dir_path in dirs:
    dir = Path(dir_path)
    files = sorted(dir.glob("*_long.csv"))
    
    if not files:
        print(f"⚠️ No *_long.csv found under {dir_path}")
        continue

    for fp in files:
        print(f"Reading: {fp}")
        df = pd.read_csv(fp)
        
        # Check that the necessary columns are present
        if not {"datetime", "station", "target"}.issubset(df.columns):
            print(f"⚠️ Skip (missing required columns): {fp.name}")
            continue
        
        # Append the dataframe to the list
        dfs.append(df)

# Concatenate all the dataframes
full_df = pd.concat(dfs, ignore_index=True)

# Sort by datetime and station
full_df = full_df.sort_values(by=["station", "datetime"])

# Save the merged data to a new CSV
full_df.to_csv(out_path, index=False)

print(f"\nDone! Merged data saved to: {out_path}, total rows: {len(full_df)}")

