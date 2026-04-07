import pandas as pd
import re

# 读取原始文件
df = pd.read_csv("nodes.csv")

# 清洗函数
def clean_station_name(name):
    name = re.sub(r"_\d+_?\d*kV$", "", name, flags=re.IGNORECASE)
    name = name.replace("_", " ")
    return name.strip()

# 生成 clean 列
df["station_clean"] = df["node_id"].apply(clean_station_name)

# 只保留这一列
df_clean = df[["station_clean"]]

# 保存
df_clean.to_csv("stations_clean_only.csv", index=False, encoding="utf-8-sig")

print("✅ 已生成 stations_clean_only.csv")
