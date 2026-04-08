# train_xgb_from_ready_parquet.py
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

def load_parts(d):
    files = sorted(glob.glob(os.path.join(d, "part_*.parquet")))
    if not files:
        raise ValueError(f"no parquet parts in {d}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def metrics(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"[{name}] mse={mse:.10f}, mae={mae:.10f}, rmse={rmse:.10f}")
    return {"mse": mse, "mae": mae, "rmse": rmse}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_df = load_parts(args.train_dir)
    val_df = load_parts(args.val_dir)
    test_df = load_parts(args.test_dir)

    feature_cols = [
        "sid_idx", "horizon", "y_hat_T",
        "lag_1", "lag_4", "lag_96", "lag_672",
        "rolling_mean_4", "rolling_std_4",
        "rolling_mean_96", "rolling_std_96",
        "hour", "minute", "day_of_week", "month", "day", "is_weekend",
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["label"]

    X_val = val_df[feature_cols]
    y_val = val_df["label"]

    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2021,
        n_jobs=8,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True
    )

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    metrics("train_label_space", y_train, pred_train)
    metrics("val_label_space", y_val, pred_val)
    metrics("test_label_space", y_test, pred_test)

    final_train = train_df["y_hat_T"].values + pred_train
    final_val = val_df["y_hat_T"].values + pred_val
    final_test = test_df["y_hat_T"].values + pred_test

    metrics("train_final", train_df["y_true"].values, final_train)
    metrics("val_final", val_df["y_true"].values, final_val)
    final_metrics = metrics("test_final", test_df["y_true"].values, final_test)
    base_metrics = metrics("test_autotimes_base", test_df["y_true"].values, test_df["y_hat_T"].values)

    model_path = os.path.join(args.out_dir, "xgb_unified.json")
    model.save_model(model_path)

    summary = {
        "feature_cols": feature_cols,
        "final_metrics": final_metrics,
        "base_metrics": base_metrics,
        "model_path": model_path,
    }
    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE] saved model to", model_path)

if __name__ == "__main__":
    main()
    '''python train_xgb_from_ready_parquet.py \
  --train_dir /root/autodl-tmp/xgb_ready_exports/你的setting_train_xgb_ready \
  --val_dir /root/autodl-tmp/xgb_ready_exports/你的setting_val_xgb_ready \
  --test_dir /root/autodl-tmp/xgb_ready_exports/你的setting_test_xgb_ready \
  --out_dir /root/autodl-tmp/xgb_runs/run1'''