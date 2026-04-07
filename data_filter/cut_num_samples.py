import pandas as pd
import numpy as np

SEQ_LEN = 672   # 历史长度
PRED_LEN = 96   # 预测长度

long = pd.read_csv("/root/autodl-tmp/datasets/load_long_with_id.csv", parse_dates=["datetime"])
long = long.sort_values(["station_id", "datetime"]).reset_index(drop=True)

X_list, Y_list, sid_list = [], [], []

for sid, g in long.groupby("station_id", sort=True):
    y = g["target"].to_numpy(dtype=np.float32)

    # 可选：每站点标准化（强烈建议；否则不同站点量纲差别很大）
    mu = y.mean()
    sigma = y.std() + 1e-6
    y = (y - mu) / sigma

    total = len(y)
    # 每个样本：x = y[i:i+SEQ_LEN], y_true = y[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN]
    max_i = total - (SEQ_LEN + PRED_LEN)
    if max_i <= 0:
        continue

    for i in range(0, max_i):
        X_list.append(y[i:i+SEQ_LEN])
        Y_list.append(y[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN])
        sid_list.append(sid)

X = np.stack(X_list)          # [N, 672]
Y = np.stack(Y_list)          # [N, 96]
sid_arr = np.array(sid_list, dtype=np.int32)

print("X:", X.shape, "Y:", Y.shape, "sid:", sid_arr.shape)
np.save("/root/autodl-tmp/datasets/x.npy", X)
np.save("/root/autodl-tmp/Y.npy", Y)
np.save("/root/autodl-tmp/sid.npy", sid_arr)
print("saved X.npy Y.npy sid.npy")
