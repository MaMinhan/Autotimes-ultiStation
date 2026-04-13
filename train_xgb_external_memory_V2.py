import os
import glob
import json
import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb


def ensure_cache_prefix_usable(cache_prefix: str, min_free_gb: float = 5.0) -> None:
    """
    检查 external memory 的 cache_prefix 是否可用。
    注意：
    - XGBoost 这里需要的是“前缀”，不是一个已存在的目录
    - 真正必须存在且可写的是 cache_prefix 的父目录
    """
    prefix_path = Path(cache_prefix)
    parent = prefix_path.parent

    parent.mkdir(parents=True, exist_ok=True)

    if prefix_path.exists() and prefix_path.is_dir():
        raise RuntimeError(
            f"cache_prefix 不能指向一个已存在的目录：{prefix_path}\n"
            f"请删除该目录，或改用别的前缀名。"
        )

    if not os.access(parent, os.W_OK):
        raise RuntimeError(f"cache 父目录不可写：{parent}")

    try:
        with tempfile.NamedTemporaryFile(dir=parent, prefix=".cache_write_test_", delete=True) as f:
            f.write(b"ok")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        raise RuntimeError(f"cache 父目录无法写入临时文件：{parent}\n原始错误：{e}") from e

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
        name: str = "data",
    ) -> None:
        self.files = sorted(glob.glob(os.path.join(parquet_dir, "part_*.parquet")))
        if not self.files:
            raise ValueError(f"No parquet parts found in {parquet_dir}")

        self.feature_cols = feature_cols
        self.label_col = label_col
        self._it = 0
        self._pass_id = 0
        self.name = name

        super().__init__(cache_prefix=cache_prefix)

    def reset(self) -> None:
        self._it = 0
        self._pass_id += 1
        print(f"[ITER-{self.name}] ===== starting full pass #{self._pass_id} =====")

    def next(self, input_data: Callable) -> bool:
        if self._it >= len(self.files):
            print(f"[ITER-{self.name}] pass #{self._pass_id} finished")
            return False

        path = self.files[self._it]
        progress = 100.0 * (self._it + 1) / len(self.files)
        print(
            f"[ITER-{self.name}] pass={self._pass_id} "
            f"reading {self._it + 1}/{len(self.files)} ({progress:.1f}%): "
            f"{os.path.basename(path)}"
        )

        df = pd.read_parquet(path)
        df = sanitize_dataframe(df, self.feature_cols, self.label_col)

        X = df[self.feature_cols]
        y = df[self.label_col]

        input_data(data=X, label=y)

        self._it += 1
        return True


def evaluate_parquet_dir_chunkwise(
    booster: xgb.Booster,
    parquet_dir: str,
    feature_cols: List[str],
    label_col: str,
    name: str,
    max_parts: Optional[int] = None,
):
    files = sorted(glob.glob(os.path.join(parquet_dir, "part_*.parquet")))
    if not files:
        raise ValueError(f"No parquet parts found in {parquet_dir}")

    if max_parts is not None:
        files = files[:max_parts]

    total_n = 0

    residual_sse = 0.0
    residual_sae = 0.0

    final_sse = 0.0
    final_sae = 0.0

    base_sse = 0.0
    base_sae = 0.0

    for i, f in enumerate(files, 1):
        print(f"[EVAL-{name}] reading {i}/{len(files)}: {os.path.basename(f)}")
        df = pd.read_parquet(f)
        df = sanitize_dataframe(df, feature_cols, label_col)

        X = df[feature_cols]
        y_residual_true = df[label_col].to_numpy()
        y_true = df["y_true"].to_numpy()
        y_hat_T = df["y_hat_T"].to_numpy()

        pred_residual = booster.predict(xgb.DMatrix(X))
        pred_final = y_hat_T + pred_residual

        residual_err = y_residual_true - pred_residual
        residual_sse += float(np.sum(residual_err ** 2))
        residual_sae += float(np.sum(np.abs(residual_err)))

        final_err = y_true - pred_final
        final_sse += float(np.sum(final_err ** 2))
        final_sae += float(np.sum(np.abs(final_err)))

        base_err = y_true - y_hat_T
        base_sse += float(np.sum(base_err ** 2))
        base_sae += float(np.sum(np.abs(base_err)))

        total_n += len(df)

    if total_n == 0:
        raise RuntimeError(f"[EVAL-{name}] no valid rows after sanitization")

    residual_metrics = {
        "mse": residual_sse / total_n,
        "mae": residual_sae / total_n,
        "rmse": (residual_sse / total_n) ** 0.5,
    }
    final_metrics = {
        "mse": final_sse / total_n,
        "mae": final_sae / total_n,
        "rmse": (final_sse / total_n) ** 0.5,
    }
    base_metrics = {
        "mse": base_sse / total_n,
        "mae": base_sae / total_n,
        "rmse": (base_sse / total_n) ** 0.5,
    }

    print(
        f"[{name}_label_space] "
        f"mse={residual_metrics['mse']:.10f}, "
        f"mae={residual_metrics['mae']:.10f}, "
        f"rmse={residual_metrics['rmse']:.10f}"
    )
    print(
        f"[{name}_final] "
        f"mse={final_metrics['mse']:.10f}, "
        f"mae={final_metrics['mae']:.10f}, "
        f"rmse={final_metrics['rmse']:.10f}"
    )
    print(
        f"[{name}_autotimes_base] "
        f"mse={base_metrics['mse']:.10f}, "
        f"mae={base_metrics['mae']:.10f}, "
        f"rmse={base_metrics['rmse']:.10f}"
    )

    return residual_metrics, final_metrics, base_metrics


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
        "--num_boost_round",
        type=int,
        default=250,
        help="boosting 轮数"
    )
    parser.add_argument(
        "--verbose_eval",
        type=int,
        default=50,
        help="每多少轮打印一次 train/val rmse"
    )

    parser.add_argument(
        "--eval_max_parts",
        type=int,
        default=None,
        help="最终分块评估时最多读多少个 part；不传则全量评估"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='XGBoost device, e.g. "cuda" or "cpu"'
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cache_root = args.cache_root or os.path.join(args.out_dir, "cache")
    os.makedirs(cache_root, exist_ok=True)

    train_cache_prefix = os.path.join(cache_root, "train_cache_file")
    val_cache_prefix = os.path.join(cache_root, "val_cache_file")
    test_cache_prefix = os.path.join(cache_root, "test_cache_file")

    print("[INFO] out_dir =", args.out_dir)
    print("[INFO] cache_root =", cache_root)
    print("[INFO] train_cache_prefix =", train_cache_prefix)
    print("[INFO] val_cache_prefix   =", val_cache_prefix)
    print("[INFO] test_cache_prefix  =", test_cache_prefix)

    ensure_cache_prefix_usable(train_cache_prefix, min_free_gb=args.min_cache_free_gb)
    ensure_cache_prefix_usable(val_cache_prefix, min_free_gb=args.min_cache_free_gb)
    ensure_cache_prefix_usable(test_cache_prefix, min_free_gb=args.min_cache_free_gb)

    if args.clear_cache:
        cleanup_old_cache_files(train_cache_prefix)
        cleanup_old_cache_files(val_cache_prefix)
        cleanup_old_cache_files(test_cache_prefix)

    raw_feature_cols = get_feature_cols()
    label_col = "residual"

    train_cols = set(read_one_part_columns(args.train_dir))
    feature_cols = [c for c in raw_feature_cols if c in train_cols]

    missing_core = [c for c in ["sid_idx", "horizon", "y_hat_T", label_col, "y_true"] if c not in train_cols]
    if missing_core:
        raise ValueError(f"Missing required columns in parquet: {missing_core}")

    print("[INFO] feature cols =", feature_cols)
    print("[INFO] label col =", label_col)

    train_it = ParquetIterator(
        parquet_dir=args.train_dir,
        feature_cols=feature_cols,
        label_col=label_col,
        cache_prefix=train_cache_prefix,
        name="train",
    )
    val_it = ParquetIterator(
        parquet_dir=args.val_dir,
        feature_cols=feature_cols,
        label_col=label_col,
        cache_prefix=val_cache_prefix,
        name="val",
    )
    test_it = ParquetIterator(
        parquet_dir=args.test_dir,
        feature_cols=feature_cols,
        label_col=label_col,
        cache_prefix=test_cache_prefix,
        name="test",
    )

    print("[STAGE] building dtrain...")
    dtrain = xgb.ExtMemQuantileDMatrix(train_it)
    print("[STAGE] dtrain ready")

    print("[STAGE] building dval...")
    dval = xgb.ExtMemQuantileDMatrix(val_it, ref=dtrain)
    print("[STAGE] dval ready")

    print("[STAGE] building dtest...")
    dtest = xgb.ExtMemQuantileDMatrix(test_it, ref=dtrain)
    print("[STAGE] dtest ready")

    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": args.device,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "rmse",
        "seed": 2021,
    }

    print("[STAGE] start xgboost training...")
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=args.verbose_eval,
    )
    print("[STAGE] training finished")

    model_path = os.path.join(args.out_dir, "xgb_extmem.json")
    booster.save_model(model_path)

    print("[STAGE] start chunkwise evaluation on val...")
    val_label_metrics, val_final_metrics, val_base_metrics = evaluate_parquet_dir_chunkwise(
        booster=booster,
        parquet_dir=args.val_dir,
        feature_cols=feature_cols,
        label_col=label_col,
        name="val",
        max_parts=args.eval_max_parts,
    )

    print("[STAGE] start chunkwise evaluation on test...")
    test_label_metrics, test_final_metrics, test_base_metrics = evaluate_parquet_dir_chunkwise(
        booster=booster,
        parquet_dir=args.test_dir,
        feature_cols=feature_cols,
        label_col=label_col,
        name="test",
        max_parts=args.eval_max_parts,
    )

    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "label_col": label_col,
                "val_label_space": val_label_metrics,
                "val_final": val_final_metrics,
                "val_autotimes_base": val_base_metrics,
                "test_label_space": test_label_metrics,
                "test_final": test_final_metrics,
                "test_autotimes_base": test_base_metrics,
                "model_path": model_path,
                "cache_root": cache_root,
                "train_cache_prefix": train_cache_prefix,
                "val_cache_prefix": val_cache_prefix,
                "test_cache_prefix": test_cache_prefix,
                "eval_max_parts": args.eval_max_parts,
                "num_boost_round": args.num_boost_round,
                "device": args.device,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[DONE] saved model to", model_path)


if __name__ == "__main__":
    main()

'''
python train_xgb_external_memory_V2.py \
  --train_dir /root/autodl-tmp/xgb_exports/CKPT_From_multi_scale_time-pt_no_prefix/no_exog/train_predictions \
  --val_dir /root/autodl-tmp/xgb_exports/CKPT_From_multi_scale_time-pt_no_prefix/no_exog/val_predictions \
  --test_dir /root/autodl-tmp/xgb_exports/CKPT_From_multi_scale_time-pt_no_prefix/no_exog/test_predictions \
  --out_dir /root/autodl-tmp/xgb_exports/CKPT_From_multi_scale_time-pt_no_prefix/no_exog/output \
  --clear_cache \
  --num_boost_round 250 \
  --verbose_eval 50 

'''