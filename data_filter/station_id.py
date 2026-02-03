import pandas as pd

long = pd.read_csv("/root/autodl-tmp/datasets/merged_FY22_24_long_cleaned.csv")
stations = sorted(long["station"].unique())
station2id = {s:i for i,s in enumerate(stations)}

long["station_id"] = long["station"].map(station2id).astype(int)
long.to_csv("load_long_with_id.csv", index=False)

print("num_stations:", len(stations))
print("example mapping:", list(station2id.items())[:5])
