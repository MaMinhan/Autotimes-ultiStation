import pandas as pd
import numpy as np

path = "/root/autodl-tmp/datasets/SelfMadeAusgridData/merged_include_id_filled.csv"
df = pd.read_csv(path)

# 只扫描数值列（尤其 target）
num = df.select_dtypes(include=[np.number])

print("nan cnt:\n", num.isna().sum().sort_values(ascending=False).head(20))
print("inf cnt:\n", np.isinf(num.to_numpy()).sum())

# 专门看 target
t = num["target"].to_numpy() if "target" in num.columns else None
if t is not None:
    print("target finite:", np.isfinite(t).all(),
          "nan:", np.isnan(t).sum(), "inf:", np.isinf(t).sum(),
          "min:", np.nanmin(t), "max:", np.nanmax(t))
