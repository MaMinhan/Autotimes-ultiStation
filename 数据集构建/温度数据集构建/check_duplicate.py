import pandas as pd

path = "/root/autodl-tmp/datasets/SelfMadeAusgridData/stations_to_SA2_SA3_SA4_2021.csv"
df = pd.read_csv(path)
df.columns = [c.strip() for c in df.columns]

print("原始行数:", len(df))
print("station_clean唯一值数量:", df["station_clean"].astype(str).str.strip().nunique())