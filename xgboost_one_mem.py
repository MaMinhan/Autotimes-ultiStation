import os
import glob
import json
import gc
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

warnings.filterwarnings("ignore")


def metrics(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"[{name}] mse={mse:.10f}, mae={mae:.10f}, rmse={rmse:.10f}")
    return {"mse": float(mse), "mae": float(mae), "rmse": float(rmse)}


def add_time_features_inplace(df: pd.DataFrame) -> None:
    """
    如果 parquet 里没有 hour/day_of_week/...，但有 target_time，
    就现场补时间特征。
    """
    if "target_time" not in df.columns:
        return

    ts = pd.to_datetime(df["target_time"], errors="coerce")

    if "hour" not in df.columns:
        df["hour"] = ts.dt.hour
    if "minute" not in df.columns:
        df["minute"] = ts.dt.minute
    if "day_of_week" not in df.columns:
        df["day_of_week"] = ts.dt.dayofweek
    if "month" not in df.columns:
        df["month"] = ts.dt.month
    if "day" not in df.columns:
        df["day"] = ts.dt.day
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(np.int8)


def optimize_dtypes_inplace(df: pd.DataFrame) -> None:
    int_cols = [
        "sid_idx", "horizon", "hour", "minute", "day_of_week",
        "month", "day", "is_weekend"
    ]
    float_cols = [
        "y_hat_T", "y_true", "label", "residual",
        "lag_1", "lag_4", "lag_96", "lag_672",
        "rolling_mean_4", "rolling_std_4",
        "rolling_mean_96", "rolling_std_96",
    ]

    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], downcast="integer")

    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], downcast="float")


def find_label_column(df: pd.DataFrame) -> str:
    if "label" in df.columns:
        return "label"
    if "residual" in df.columns:
        return "residual"
    raise ValueError("Neither 'label' nor 'residual' exists in dataframe.")


def get_feature_cols(df: pd.DataFrame):
    """
    优先使用 xgb_ready 完整特征；
    如果没有 lag/rolling，则退化为基础版特征。
    """
    preferred = [
        "sid_idx",
        "horizon",
        "y_hat_T",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "lag_1",
        "lag_4",
        "lag_96",
        "lag_672",
        "rolling_mean_4",
        "rolling_std_4",
        "rolling_mean_96",
        "rolling_std_96",
    ]

    fallback = [
        "sid_idx",
        "horizon",
        "y_hat_T",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "minute",
        "day",
    ]

    if all(c in df.columns for c in preferred):
        return preferred

    cols = [c for c in fallback if c in df.columns]
    missing_core = [c for c in ["sid_idx", "horizon", "y_hat_T"] if c not in cols]
    if missing_core:
        raise ValueError(f"Missing required core feature cols: {missing_core}")
    return cols


def maybe_sample_df(
    df: pd.DataFrame,
    sample_frac: float = 1.0,
    max_rows: int = 0,
    random_state: int = 2021,
    tag: str = ""
) -> pd.DataFrame:
    """
    小样本微调/试跑：
    1) 先按比例采样
    2) 再按最大行数截断
    """
    original_n = len(df)

    if sample_frac is not None and 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)

    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state)

    df = df.reset_index(drop=True)

    if tag:
        print(f"[SAMPLE] {tag}: {original_n} -> {len(df)} rows")

    return df


def build_lightweight_training_df_from_folder(
    folder: str,
    max_parts=None,
    sample_frac: float = 1.0,
    max_rows: int = 0,
    random_state: int = 2021,
    tag: str = ""
):
    """
    每个 parquet part 都按“全列读取”；
    但读完立即压成训练需要的轻量列，减少内存占用。
    """
    files = sorted(glob.glob(os.path.join(folder, "part_*.parquet")))
    if not files:
        raise ValueError(f"No parquet parts found in {folder}")

    if max_parts is not None:
        files = files[:max_parts]

    processed_parts = []
    final_feature_cols = None
    final_label_col = None

    print(f"[LOAD] folder={folder}")
    print(f"[LOAD] parts={len(files)}")

    for i, f in enumerate(files, 1):
        print(f"  reading full parquet {i}/{len(files)}: {os.path.basename(f)}")

        # 按你的要求：读所有列
        df = pd.read_parquet(f)

        add_time_features_inplace(df)
        label_col = find_label_column(df)
        feature_cols = get_feature_cols(df)

        keep_cols = list(dict.fromkeys(feature_cols + [label_col, "y_true", "y_hat_T"]))
        slim = df[keep_cols].copy()
        optimize_dtypes_inplace(slim)

        if final_feature_cols is None:
            final_feature_cols = feature_cols
            final_label_col = label_col
            print("[INFO] feature cols =", final_feature_cols)
            print("[INFO] label col    =", final_label_col)

        processed_parts.append(slim)

        del df, slim
        gc.collect()

    print("[LOAD] concatenating lightweight parts...")
    out_df = pd.concat(processed_parts, ignore_index=True)
    del processed_parts
    gc.collect()

    print(f"[LOAD DONE] shape={out_df.shape}")
    mem_mb = out_df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"[LOAD DONE] approx memory={mem_mb:.2f} MB")

    out_df = maybe_sample_df(
        out_df,
        sample_frac=sample_frac,
        max_rows=max_rows,
        random_state=random_state,
        tag=tag
    )

    mem_mb_after = out_df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"[LOAD DONE AFTER SAMPLE] shape={out_df.shape}")
    print(f"[LOAD DONE AFTER SAMPLE] approx memory={mem_mb_after:.2f} MB")

    return out_df, final_feature_cols, final_label_col


def dataframe_to_numpy_and_release(df, feature_cols, label_col, tag=""):
    """
    把 DataFrame 转成 numpy float32，并尽量释放 pandas 内存。
    """
    X = df[feature_cols]
    y = df[label_col]

    y_true_np = df["y_true"].to_numpy(dtype=np.float32, copy=False)
    y_hat_np = df["y_hat_T"].to_numpy(dtype=np.float32, copy=False)

    print(f"\n========== converting {tag} to numpy float32 ==========")
    X_np = X.to_numpy(dtype=np.float32, copy=False)
    y_np = y.to_numpy(dtype=np.float32, copy=False)

    del X, y, df
    gc.collect()

    return X_np, y_np, y_true_np, y_hat_np


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    # 只读前几个 part，适合先试跑
    parser.add_argument("--max_train_parts", type=int, default=None)
    parser.add_argument("--max_val_parts", type=int, default=None)
    parser.add_argument("--max_test_parts", type=int, default=None)

    # 小量数据微调/试跑
    parser.add_argument("--train_sample_frac", type=float, default=1.0)
    parser.add_argument("--val_sample_frac", type=float, default=1.0)
    parser.add_argument("--test_sample_frac", type=float, default=1.0)

    parser.add_argument("--train_max_rows", type=int, default=0)
    parser.add_argument("--val_max_rows", type=int, default=0)
    parser.add_argument("--test_max_rows", type=int, default=0)

    # xgboost 参数
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--max_bin", type=int, default=128)
    parser.add_argument("--random_state", type=int, default=2021)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ========== 1) load train ==========
    print("\n========== loading train ==========")
    train_df, feature_cols, label_col = build_lightweight_training_df_from_folder(
        args.train_dir,
        max_parts=args.max_train_parts,
        sample_frac=args.train_sample_frac,
        max_rows=args.train_max_rows,
        random_state=args.random_state,
        tag="train"
    )

    # ========== 2) load val ==========
    print("\n========== loading val ==========")
    val_df, val_feature_cols, val_label_col = build_lightweight_training_df_from_folder(
        args.val_dir,
        max_parts=args.max_val_parts,
        sample_frac=args.val_sample_frac,
        max_rows=args.val_max_rows,
        random_state=args.random_state,
        tag="val"
    )

    if feature_cols != val_feature_cols:
        raise ValueError(
            f"Train/Val feature cols mismatch.\n"
            f"train={feature_cols}\nval={val_feature_cols}"
        )
    if label_col != val_label_col:
        raise ValueError(
            f"Train/Val label col mismatch: train={label_col}, val={val_label_col}"
        )

    X_train_np, y_train_np, train_y_true, train_y_hat = dataframe_to_numpy_and_release(
        train_df, feature_cols, label_col, tag="train"
    )
    X_val_np, y_val_np, val_y_true, val_y_hat = dataframe_to_numpy_and_release(
        val_df, feature_cols, label_col, tag="val"
    )

    print("\n========== training xgboost ==========")
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        tree_method="hist",
        max_bin=args.max_bin,
    )

    model.fit(
        X_train_np,
        y_train_np,
        eval_set=[(X_val_np, y_val_np)],
        verbose=True
    )

    print("\n========== predicting train/val ==========")
    pred_train = model.predict(X_train_np)
    pred_val = model.predict(X_val_np)

    train_label_space = metrics("train_label_space", y_train_np, pred_train)
    val_label_space = metrics("val_label_space", y_val_np, pred_val)

    final_train = train_y_hat + pred_train
    final_val = val_y_hat + pred_val

    train_final = metrics("train_final", train_y_true, final_train)
    val_final = metrics("val_final", val_y_true, final_val)

    # 释放 train/val numpy
    del X_train_np, y_train_np, X_val_np, y_val_np
    del pred_train, pred_val, final_train, final_val
    del train_y_true, train_y_hat, val_y_true, val_y_hat
    gc.collect()

    # ========== 3) load test ==========
    print("\n========== loading test ==========")
    test_df, test_feature_cols, test_label_col = build_lightweight_training_df_from_folder(
        args.test_dir,
        max_parts=args.max_test_parts,
        sample_frac=args.test_sample_frac,
        max_rows=args.test_max_rows,
        random_state=args.random_state,
        tag="test"
    )

    if feature_cols != test_feature_cols:
        raise ValueError(
            f"Train/Test feature cols mismatch.\n"
            f"train={feature_cols}\ntest={test_feature_cols}"
        )
    if label_col != test_label_col:
        raise ValueError(
            f"Train/Test label col mismatch: train={label_col}, test={test_label_col}"
        )

    X_test_np, y_test_np, test_y_true, test_y_hat = dataframe_to_numpy_and_release(
        test_df, feature_cols, label_col, tag="test"
    )

    print("\n========== predicting test ==========")
    pred_test = model.predict(X_test_np)

    test_label_space = metrics("test_label_space", y_test_np, pred_test)

    final_test = test_y_hat + pred_test
    test_final = metrics("test_final", test_y_true, final_test)
    base_test = metrics("test_autotimes_base", test_y_true, test_y_hat)

    # 保存模型
    model_path = os.path.join(args.out_dir, "xgb_unified.json")
    model.save_model(model_path)

    out_test = pd.DataFrame({
        "y_true": test_y_true,
        "y_hat_T": test_y_hat,
        label_col: y_test_np,
        "xgb_residual_hat": pred_test,
        "final_pred": final_test,
    })
    out_test_path = os.path.join(args.out_dir, "xgb_test_predictions.parquet")
    out_test.to_parquet(out_test_path, index=False)

    summary = {
        "feature_cols": feature_cols,
        "label_col": label_col,
        "train_label_space": train_label_space,
        "val_label_space": val_label_space,
        "test_label_space": test_label_space,
        "train_final": train_final,
        "val_final": val_final,
        "test_final": test_final,
        "test_autotimes_base": base_test,
        "model_path": model_path,
        "out_test_path": out_test_path,
        "sampling": {
            "max_train_parts": args.max_train_parts,
            "max_val_parts": args.max_val_parts,
            "max_test_parts": args.max_test_parts,
            "train_sample_frac": args.train_sample_frac,
            "val_sample_frac": args.val_sample_frac,
            "test_sample_frac": args.test_sample_frac,
            "train_max_rows": args.train_max_rows,
            "val_max_rows": args.val_max_rows,
            "test_max_rows": args.test_max_rows,
        },
        "xgb_params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "n_jobs": args.n_jobs,
            "max_bin": args.max_bin,
            "tree_method": "hist",
        }
    }

    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== done ==========")
    print("model saved to:", model_path)
    print("test predictions saved to:", out_test_path)


if __name__ == "__main__":
    main()
'''python xgboost2.py \
  --train_dir /root/autodl-tmp/xgb_exports/CKPT_From_多尺度_time-pt_no_prefix/with_exog/train \
  --val_dir /root/autodl-tmp/xgb_exports/CKPT_From_多尺度_time-pt_no_prefix/with_exog/val \
  --test_dir /root/autodl-tmp/xgb_exports/CKPT_From_多尺度_time-pt_no_prefix/with_exog/test \
  --out_dir /root/autodl-tmp/xgb_exports/CKPT_From_多尺度_time-pt_no_prefix/with_exog/output \
  --max_train_parts 100 \
  --max_val_parts 20 \
  --max_test_parts 20 \
  --train_sample_frac 0.1 \
  --val_sample_frac 0.3 \
  --test_sample_frac 0.3 \
  --n_estimators 80 \
  --max_depth 4 \
  --n_jobs 4'''