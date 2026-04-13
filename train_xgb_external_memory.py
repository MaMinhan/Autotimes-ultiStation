import os
import glob
import json
import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

def ensure_cache_prefix_usable(cache_prefix: str, min_free_gb: float = 5.0) -> None:
    """
    检查 external memory 的 cache_prefix 是否可用。
    注意：
    - XGBoost 这里需要的是“前缀”，不是一个已存在的目录
    - 真正必须存在且可写的是 cache_prefix 的父目录
    """
    prefix_path = Path(cache_prefix)
    parent = prefix_path.parent

    # 1. 父目录不存在则创建
    parent.mkdir(parents=True, exist_ok=True)

    # 2. prefix 本身不能是目录
    if prefix_path.exists() and prefix_path.is_dir():
        raise RuntimeError(
            f"cache_prefix 不能指向一个已存在的目录：{prefix_path}\n"
            f"请删除该目录，或改用别的前缀名。"
        )

    # 3. 父目录必须可写
    if not os.access(parent, os.W_OK):
        raise RuntimeError(f"cache 父目录不可写：{parent}")

    # 4. 尝试真实写一个临时文件，验证可写性
    try:
        with tempfile.NamedTemporaryFile(dir=parent, prefix=".cache_write_test_", delete=True) as f:
            f.write(b"ok")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        raise RuntimeError(f"cache 父目录无法写入临时文件：{parent}\n原始错误：{e}") from e

    # 5. 检查磁盘空间
    usage = shutil.disk_usage(parent)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"cache 父目录剩余空间不足：{parent}\n"
            f"当前剩余 {free_gb:.2f} GB，低于要求的 {min_free_gb:.2f} GB"
        )

    print(f"[CACHE CHECK] OK: {cache_prefix}")
    print(f"[CACHE CHECK] parent={parent}")
    print(f"[CACHE CHECK] free={free_gb:.2f} GB")


def cleanup_old_cache_files(cache_prefix: str) -> None:
    """
    删除同一 prefix 遗留的旧 cache 文件，避免干扰本次运行。
    """
    prefix_path = Path(cache_prefix)
    parent = prefix_path.parent
    pattern = prefix_path.name + "*"

    removed = 0
    if parent.exists():
        for p in parent.glob(pattern):
            if p.is_file():
                p.unlink()
                removed += 1

    print(f"[CACHE CLEAN] removed {removed} stale files for prefix={cache_prefix}")

def get_feature_cols() -> List[str]:
    return [
        "sid_idx",
        "horizon",
        "y_hat_T",

        "prefix_cal_month_norm",
        "prefix_cal_season_spring",
        "prefix_cal_season_summer",
        "prefix_cal_season_autumn",
        "prefix_cal_season_winter",
        "prefix_cal_dow_0",
        "prefix_cal_dow_1",
        "prefix_cal_dow_2",
        "prefix_cal_dow_3",
        "prefix_cal_dow_4",
        "prefix_cal_dow_5",
        "prefix_cal_dow_6",
        "prefix_cal_daytype_workday",
        "prefix_cal_daytype_weekend",
        "prefix_cal_daytype_holiday",
        "prefix_cal_holidayrel_pre",
        "prefix_cal_holidayrel_cur",
        "prefix_cal_holidayrel_post",

        "prefix_social_pop_density_per_sqkm_norm",
        "prefix_social_young_ratio_0_14_norm",
        "prefix_social_working_ratio_15_64_norm",
        "prefix_social_elderly_ratio_65_plus_norm",
        "prefix_social_private_dwelling_ratio_norm",
        "prefix_social_other_dwelling_ratio_norm",

        "temp_last",
        "temp_mean_4",
        "temp_mean_96",
        "temp_std_96",
    ]


def read_one_part_columns(parquet_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(parquet_dir, "part_*.parquet")))
    if not files:
        raise ValueError(f"No parquet parts found in {parquet_dir}")
    df = pd.read_parquet(files[0])
    cols = df.columns.tolist()
    del df
    return cols


def sanitize_dataframe(df: pd.DataFrame, feature_cols: List[str], label_col: str) -> pd.DataFrame:
    need_cols = feature_cols + [label_col, "y_true", "y_hat_T"]
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=need_cols).reset_index(drop=True)
    return df


class ParquetIterator(xgb.DataIter):
    def __init__(
        self,
        parquet_dir: str,
        feature_cols: List[str],
        label_col: str,
        cache_prefix: str,
    ) -> None:
        self.files = sorted(glob.glob(os.path.join(parquet_dir, "part_*.parquet")))
        if not self.files:
            raise ValueError(f"No parquet parts found in {parquet_dir}")

        self.feature_cols = feature_cols
        self.label_col = label_col
        self._it = 0

        super().__init__(cache_prefix=cache_prefix)

    def reset(self) -> None:
        self._it = 0

    def next(self, input_data: Callable) -> bool:
        if self._it >= len(self.files):
            return False

        path = self.files[self._it]
        print(f"[ITER] reading {self._it + 1}/{len(self.files)}: {os.path.basename(path)}")

        df = pd.read_parquet(path)
        df = sanitize_dataframe(df, self.feature_cols, self.label_col)

        X = df[self.feature_cols]
        y = df[self.label_col]

        input_data(data=X, label=y)

        self._it += 1
        return True


def load_eval_subset(parquet_dir: str, feature_cols: List[str], label_col: str, max_parts: int | None = None):
    files = sorted(glob.glob(os.path.join(parquet_dir, "part_*.parquet")))
    if not files:
        raise ValueError(f"No parquet parts found in {parquet_dir}")
    if max_parts is not None:
        files = files[:max_parts]

    dfs = []
    print(f"[LOAD] parts={len(files)}")
    for i, f in enumerate(files, 1):
        print(f"  reading full parquet {i}/{len(files)}: {os.path.basename(f)}")
        df = pd.read_parquet(f)
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    full = sanitize_dataframe(full, feature_cols, label_col)

    X = full[feature_cols]
    y = full[label_col]
    return full, X, y


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

    parser.add_argument(
        "--cache_root",
        type=str,
        default=None,
        help="external memory cache 的父目录；不传则默认使用 out_dir/cache"
    )
    parser.add_argument(
        "--min_cache_free_gb",
        type=float,
        default=5.0,
        help="cache 目录至少需要的剩余空间（GB）"
    )
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="启动前清理旧的 cache 文件"
    )
    parser.add_argument(
        "--eval_max_parts",
        type=int,
        default=500,
        help="验证/测试时最多读多少个 part；全量评估可设更大"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 只需要一个 cache 父目录
    cache_root = args.cache_root or os.path.join(args.out_dir, "cache")
    os.makedirs(cache_root, exist_ok=True)

    # 注意：这里是“前缀”，不是目录
    # 用 *_file 避开你已经手动创建的 train_cache / val_cache / test_cache 目录
    train_cache_prefix = os.path.join(cache_root, "train_cache_file")
    val_cache_prefix = os.path.join(cache_root, "val_cache_file")
    test_cache_prefix = os.path.join(cache_root, "test_cache_file")

    print("[INFO] out_dir =", args.out_dir)
    print("[INFO] cache_root =", cache_root)
    print("[INFO] train_cache_prefix =", train_cache_prefix)
    print("[INFO] val_cache_prefix   =", val_cache_prefix)
    print("[INFO] test_cache_prefix  =", test_cache_prefix)

    # 启动时检查 cache prefix 可用性
    ensure_cache_prefix_usable(train_cache_prefix, min_free_gb=args.min_cache_free_gb)
    ensure_cache_prefix_usable(val_cache_prefix, min_free_gb=args.min_cache_free_gb)
    ensure_cache_prefix_usable(test_cache_prefix, min_free_gb=args.min_cache_free_gb)

    if args.clear_cache:
        cleanup_old_cache_files(train_cache_prefix)
        cleanup_old_cache_files(val_cache_prefix)
        cleanup_old_cache_files(test_cache_prefix)

    raw_feature_cols = get_feature_cols()
    label_col = "residual"

    # 只保留 train parquet 里真实存在的列
    train_cols = set(read_one_part_columns(args.train_dir))
    feature_cols = [c for c in raw_feature_cols if c in train_cols]

    missing_core = [c for c in ["sid_idx", "horizon", "y_hat_T", label_col, "y_true"] if c not in train_cols]
    if missing_core:
        raise ValueError(f"Missing required columns in parquet: {missing_core}")

    print("[INFO] feature cols =", feature_cols)
    print("[INFO] label col =", label_col)

    # 当前只有训练集使用 external memory
    train_it = ParquetIterator(
        parquet_dir=args.train_dir,
        feature_cols=feature_cols,
        label_col=label_col,
        cache_prefix=train_cache_prefix,
    )

    dtrain = xgb.ExtMemQuantileDMatrix(train_it)

    # val / test 仍然走内存评估
    val_df, X_val, y_val = load_eval_subset(
        args.val_dir, feature_cols, label_col, max_parts=args.eval_max_parts
    )
    test_df, X_test, y_test = load_eval_subset(
        args.test_dir, feature_cols, label_col, max_parts=args.eval_max_parts
    )

    dval = xgb.QuantileDMatrix(X_val, y_val, ref=dtrain)

    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "rmse",
        "seed": 2021,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=250,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=50,
    )

    model_path = os.path.join(args.out_dir, "xgb_extmem.json")
    booster.save_model(model_path)

    pred_val = booster.predict(xgb.QuantileDMatrix(X_val, ref=dtrain))
    pred_test = booster.predict(xgb.QuantileDMatrix(X_test, ref=dtrain))

    metrics("val_label_space", y_val, pred_val)
    metrics("test_label_space", y_test, pred_test)

    final_val = val_df["y_hat_T"].values + pred_val
    final_test = test_df["y_hat_T"].values + pred_test

    val_final = metrics("val_final", val_df["y_true"].values, final_val)
    test_final = metrics("test_final", test_df["y_true"].values, final_test)
    base_test = metrics("test_autotimes_base", test_df["y_true"].values, test_df["y_hat_T"].values)

    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "label_col": label_col,
                "val_final": val_final,
                "test_final": test_final,
                "test_autotimes_base": base_test,
                "model_path": model_path,
                "eval_max_parts": args.eval_max_parts,
                "cache_root": cache_root,
                "train_cache_prefix": train_cache_prefix,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[DONE] saved model to", model_path)

if __name__ == "__main__":
    main()
    '''
python train_xgb_external_memory.py \
  --train_dir /root/autodl-tmp/xgb_exports/CKPT_From_multi_scale_time-pt_no_prefix/with_exog/train \
  --val_dir /root/autodl-tmp/xgb_exports/CKPT_From_multi_scale_time-pt_no_prefix/with_exog/val \
  --test_dir /root/autodl-tmp/xgb_exports/CKPT_From_multi_scale_time-pt_no_prefix/with_exog/test \
  --out_dir /root/autodl-tmp/xgb_exports/CKPT_From_multi_scale_time-pt_no_prefix/with_exog/output 
'''