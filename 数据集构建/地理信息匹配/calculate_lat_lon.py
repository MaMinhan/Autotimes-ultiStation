import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# 读取你刚刚生成的文件
df = pd.read_csv("stations_name_cleaned.csv")

# 初始化地理编码器
geolocator = Nominatim(user_agent="ausgrid_locator")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# 查询经纬度
def get_lat_lon(name):
    try:
        location = geocode(f"{name}, NSW, Australia")
        if location:
            return pd.Series([location.latitude, location.longitude])
        else:
            return pd.Series([None, None])
    except:
        return pd.Series([None, None])

# 执行查询
df[["lat", "lon"]] = df["station_clean"].apply(get_lat_lon)

# 保存结果
df.to_csv("stations_with_latlon.csv", index=False, encoding="utf-8-sig")

print("✅ 已生成 stations_with_latlon.csv")
