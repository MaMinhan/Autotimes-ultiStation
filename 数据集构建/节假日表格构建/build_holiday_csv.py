import pandas as pd
import holidays

start_date = "2021-05-01"
end_date = "2024-04-30"

# 生成按天日期范围
days = pd.date_range(start=start_date, end=end_date, freq="D")

# NSW 公共节假日
nsw_holidays = holidays.country_holidays("AU", subdiv="NSW")

df_day = pd.DataFrame({
    "date": days
})
df_day["is_holiday"] = df_day["date"].dt.date.map(lambda d: d in nsw_holidays)
df_day["holiday_name"] = df_day["date"].dt.date.map(lambda d: nsw_holidays.get(d, ""))

print(df_day.head())
print(df_day[df_day["is_holiday"]].head(20))

df_day.to_csv("/root/autodl-tmp/datasets/SelfMadeAusgridData/节假日表格/holiday.csv", index=False, encoding="utf-8-sig")