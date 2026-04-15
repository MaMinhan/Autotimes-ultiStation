import os
import glob
import json
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb



TIME_SHIFT_CANDIDATES_MINUTES = [0, 15, -15, 30, -30, 45, -45, 60, -60]


# =========================
# 基础工具
# =========================
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

def build_scaler_from_raw_csv(
    raw_csv_path: str,
    train_ratio: float = 0.7,
):
    """
    直接复刻 Dataset_MultiStation_Custom.__read_data__ 里的缩放逻辑：
    - 用原始 long-format csv
    - 构造全局唯一时间轴
    - 每个站点 reindex 到全局 dt
    - 对 target 做插值/ffill/bfill
    - 按 train 区间计算每个站点的 mu / sd
    """
    df = pd.read_csv(raw_csv_path, usecols=["datetime", "station", "target"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "station", "target"])
    df["station"] = df["station"].astype(str)

    # 全局时间轴
    dt = (
        df["datetime"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    T = len(dt)
    dt_index = pd.Index(dt)

    stations = sorted(df["station"].unique().tolist())
    sid2idx = {s: i for i, s in enumerate(stations)}

    N = len(stations)
    Y = np.full((N, T), np.nan, dtype=np.float32)

    for s in stations:
        sid_idx = sid2idx[s]
        sdf = df[df["station"] == s].sort_values("datetime")

        # 同站点同时间重复 -> 均值
        sdf = sdf.groupby("datetime", as_index=True).mean(numeric_only=True)

        # reindex 到全局 dt
        sdf = sdf.reindex(dt_index)

        # 和 Dataset_MultiStation_Custom 保持一致
        ss = sdf["target"].astype("float32")
        ss = ss.interpolate(limit_direction="both")
        ss = ss.ffill().bfill()
        sdf["target"] = ss

        Y[sid_idx, :] = sdf["target"].to_numpy(dtype=np.float32)

    num_train = int(T * float(train_ratio))
    train_slice = slice(0, num_train)

    mu = np.mean(Y[:, train_slice], axis=1, keepdims=True)
    sd = np.std(Y[:, train_slice], axis=1, keepdims=True)
    sd = np.maximum(sd, 1e-6)

    return {
        "dt": dt,
        "stations": stations,
        "sid2idx": sid2idx,
        "y_mu": mu.astype(np.float32),
        "y_sd": sd.astype(np.float32),
    }
def read_one_part_columns(parquet_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(parquet_dir, "part_*.parquet")))
    if not files:
        raise ValueError(f"No parquet parts found in {parquet_dir}")
    df = pd.read_parquet(files[0])
    cols = df.columns.tolist()
    del df
    return cols


def sanitize_parquet_df(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    needed = feature_cols + ["y_hat_T", "target_time"]
    if "y_true" in df.columns:
        needed.append("y_true")
    keep_cols = [c for c in needed if c in df.columns]

    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=keep_cols).reset_index(drop=True)
    df["target_time"] = pd.to_datetime(df["target_time"])
    return df


def inverse_transform_by_sid(values: np.ndarray, sid_idx_arr: np.ndarray, y_mu: np.ndarray, y_sd: np.ndarray) -> np.ndarray:
    """
    values: [N,]
    sid_idx_arr: [N,]
    y_mu/y_sd: [N_station, 1]
    """
    sid_idx_arr = sid_idx_arr.astype(int)
    mu = y_mu[sid_idx_arr, 0]
    sd = y_sd[sid_idx_arr, 0]
    return values * sd + mu


# =========================
# 原始 CSV / 站点映射
# =========================
def load_ground_truth_csv(csv_path: str) -> pd.DataFrame:
    """
    原始电力数据固定格式：
    datetime,station,station_id,target,miss_target
    """
    df = pd.read_csv(csv_path)

    required_cols = ["datetime", "station", "station_id", "target"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"raw csv missing required columns: {missing}")

    df = df[["datetime", "station", "station_id", "target"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    dup_cnt = df.duplicated(subset=["station_id", "datetime"]).sum()
    if dup_cnt > 0:
        print(f"[WARN] raw csv has {dup_cnt} duplicated (station_id, datetime) rows, aggregating target by mean")
        df = df.groupby(["station_id", "datetime"], as_index=False)["target"].mean()
    else:
        print("[INFO] raw csv keys are unique on (station_id, datetime)")

    return df


def load_station_map_csv(station_map_csv: str) -> pd.DataFrame:
    df = pd.read_csv(station_map_csv)

    required_cols = ["station", "station_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"station_map.csv missing required columns: {missing}")

    df = df[["station", "station_id"]].drop_duplicates().copy()
    return df


def attach_station_info_from_map(
    parquet_df: pd.DataFrame,
    station_map_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    parquet 里用 sid_idx，对应 station_map.csv 的 station_id
    """
    df = parquet_df.copy()

    if "sid_idx" not in df.columns:
        raise ValueError("parquet must contain sid_idx")

    map_df = station_map_df.rename(
        columns={
            "station_id": "_raw_station_id",
            "station": "_raw_station_name",
        }
    )

    before_n = len(df)
    df = df.merge(
        map_df,
        how="left",
        left_on="sid_idx",
        right_on="_raw_station_id",
        validate="many_to_one",
    )
    miss_n = df["_raw_station_id"].isna().sum()
    print(f"[INFO] station mapping: before={before_n}, missing_station_map={miss_n}")

    df = df.dropna(subset=["_raw_station_id"]).reset_index(drop=True)
    df["_raw_station_id"] = df["_raw_station_id"].astype(int)
    return df


# =========================
# 时间对齐
# =========================
def try_merge_with_shift(
    parquet_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    shift_minutes: int,
) -> pd.DataFrame:
    df = parquet_df.copy()
    df["_merge_time_shifted"] = df["target_time"] + pd.to_timedelta(shift_minutes, unit="m")

    before_n = len(df)

    merged = df.merge(
        gt_df,
        how="left",
        left_on=["_raw_station_id", "_merge_time_shifted"],
        right_on=["station_id", "datetime"],
        validate="many_to_one",
    )

    merged = merged.rename(columns={"target": "y_true_csv"})
    merged["_shift_minutes"] = shift_minutes

    miss_n = merged["y_true_csv"].isna().sum()
    matched_n = before_n - miss_n

    print(
        f"[MERGE TRY] shift={shift_minutes:+d} min | "
        f"before={before_n}, matched={matched_n}, missed={miss_n}"
    )
    return merged


def merge_with_ground_truth_no_shift(
    parquet_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    station_map_df: pd.DataFrame,
) -> pd.DataFrame:
    parquet_df = attach_station_info_from_map(parquet_df, station_map_df)

    before_n = len(parquet_df)

    merged = parquet_df.merge(
        gt_df,
        how="left",
        left_on=["_raw_station_id", "target_time"],
        right_on=["station_id", "datetime"],
        validate="many_to_one",
    )

    merged = merged.rename(columns={"target": "y_true_csv"})

    miss_n = merged["y_true_csv"].isna().sum()
    matched_n = before_n - miss_n

    print(
        f"[MERGE FIXED] shift=+0 min | "
        f"before={before_n}, matched={matched_n}, missed={miss_n}"
    )

    merged = merged.dropna(subset=["y_true_csv"]).reset_index(drop=True)
    return merged


def print_alignment_check_scaled_and_raw(
    name: str,
    merged: pd.DataFrame,
    y_mu: np.ndarray,
    y_sd: np.ndarray,
) -> None:
    """
    同时打印：
    1) parquet scaled y_true vs csv raw（这个通常会差很大，只做提示）
    2) parquet inverse-transformed y_true vs csv raw（这个才应该接近）
    """
    if "y_true" not in merged.columns or len(merged) == 0:
        print(f"[CHECK-{name}] parquet has no y_true column or merged is empty")
        return

    sid_idx_arr = merged["sid_idx"].astype(int).to_numpy()

    diff_scaled = np.abs(merged["y_true"].to_numpy() - merged["y_true_csv"].to_numpy())
    y_true_raw_from_parquet = inverse_transform_by_sid(
        merged["y_true"].to_numpy(),
        sid_idx_arr,
        y_mu,
        y_sd
    )
    diff_raw = np.abs(y_true_raw_from_parquet - merged["y_true_csv"].to_numpy())

    print(
        f"[CHECK-{name}] parquet_scaled_y_true vs csv_y_true: "
        f"mean_abs_diff={diff_scaled.mean():.6f}, "
        f"median_abs_diff={np.median(diff_scaled):.6f}, "
        f"max_abs_diff={diff_scaled.max():.6f}"
    )
    print(
        f"[CHECK-{name}] parquet_inverse_y_true vs csv_y_true: "
        f"mean_abs_diff={diff_raw.mean():.6f}, "
        f"median_abs_diff={np.median(diff_raw):.6f}, "
        f"max_abs_diff={diff_raw.max():.6f}"
    )


# =========================
# scaler 读取
# =========================



# =========================
# 评估函数
# =========================
def evaluate_parquet_dir_against_csv(
    booster: xgb.Booster,
    parquet_dir: str,
    feature_cols: List[str],
    gt_df: pd.DataFrame,
    station_map_df: pd.DataFrame,
    y_mu: np.ndarray,
    y_sd: np.ndarray,
    name: str,
    max_parts: Optional[int] = None,
):
    files = sorted(glob.glob(os.path.join(parquet_dir, "part_*.parquet")))
    if not files:
        raise ValueError(f"No parquet parts found in {parquet_dir}")

    if max_parts is not None:
        files = files[:max_parts]

    final_sse = 0.0
    final_sae = 0.0

    base_sse = 0.0
    base_sae = 0.0

    total_n = 0

    for i, f in enumerate(files, 1):
        print(f"[EVAL-{name}] reading {i}/{len(files)}: {os.path.basename(f)}")
        df = pd.read_parquet(f)
        df = sanitize_parquet_df(df, feature_cols)
        merged = merge_with_ground_truth_no_shift(
            parquet_df=df,
            gt_df=gt_df,
            station_map_df=station_map_df,
        )
        if len(merged) == 0:
            continue

        print_alignment_check_scaled_and_raw(name, merged, y_mu, y_sd)

        sid_idx_arr = merged["sid_idx"].astype(int).to_numpy()

        y_true_raw_from_parquet = inverse_transform_by_sid(
            merged["y_true"].to_numpy(),
            sid_idx_arr,
            y_mu,
            y_sd
        )

        align_abs_diff = np.abs(y_true_raw_from_parquet - merged["y_true_csv"].to_numpy())
        keep_mask = align_abs_diff <= 1e-3

        print(
            f"[ALIGN FILTER-{name}] keep={keep_mask.sum()}/{len(keep_mask)}, "
            f"drop={(~keep_mask).sum()}, "
            f"keep_ratio={keep_mask.mean():.6f}"
        )

        merged = merged.loc[keep_mask].reset_index(drop=True)

        if len(merged) == 0:
            continue

        # 关键修复：过滤后重新取特征和 sid
        X = merged[feature_cols]
        sid_idx_arr = merged["sid_idx"].astype(int).to_numpy()

        # parquet / xgb 输出在 scaled 空间
        y_hat_T_scaled = merged["y_hat_T"].to_numpy()
        pred_residual_scaled = booster.predict(xgb.DMatrix(X))
        pred_final_scaled = y_hat_T_scaled + pred_residual_scaled

        # inverse transform 到原始负荷空间
        y_hat_T_raw = inverse_transform_by_sid(y_hat_T_scaled, sid_idx_arr, y_mu, y_sd)
        pred_final_raw = inverse_transform_by_sid(pred_final_scaled, sid_idx_arr, y_mu, y_sd)

        # 原始 CSV 真值
        # 原始 CSV 真值
        y_true_raw = merged["y_true_csv"].to_numpy()

        # 组一个评估表
        eval_df = merged.copy()
        eval_df["pred_final_raw"] = pred_final_raw
        eval_df["y_hat_T_raw"] = y_hat_T_raw
        eval_df["y_true_raw"] = y_true_raw

        # 保证时间列是 datetime
        eval_df["target_time"] = pd.to_datetime(eval_df["target_time"])
        eval_df["forecast_start_time"] = pd.to_datetime(eval_df["forecast_start_time"])

        # 对同一个 (sid_idx, target_time)，只保留 forecast_start_time 最晚的一条
        # 若同时想更稳一点，也可再按 horizon 升序辅助排序
        before_n = len(eval_df)

        eval_df = eval_df.sort_values(
            ["sid_idx", "target_time", "forecast_start_time", "horizon"],
            ascending=[True, True, False, True]
        ).drop_duplicates(
            subset=["sid_idx", "target_time"],
            keep="first"
        ).reset_index(drop=True)

        after_n = len(eval_df)
        print(f"[DEDUP-{name}] before={before_n}, after={after_n}, keep_ratio={after_n / before_n:.6f}")

        # 用聚合后的唯一点评估
        y_true_raw_u = eval_df["y_true_raw"].to_numpy()
        pred_final_raw_u = eval_df["pred_final_raw"].to_numpy()
        y_hat_T_raw_u = eval_df["y_hat_T_raw"].to_numpy()

        final_err = y_true_raw_u - pred_final_raw_u
        final_sse += float(np.sum(final_err ** 2))
        final_sae += float(np.sum(np.abs(final_err)))

        base_err = y_true_raw_u - y_hat_T_raw_u
        base_sse += float(np.sum(base_err ** 2))
        base_sae += float(np.sum(np.abs(base_err)))

        total_n += len(eval_df)
    if total_n == 0:
        raise RuntimeError(f"[EVAL-{name}] no matched rows between parquet and original csv")

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
        f"[{name}_final_vs_csv] "
        f"mse={final_metrics['mse']:.10f}, "
        f"mae={final_metrics['mae']:.10f}, "
        f"rmse={final_metrics['rmse']:.10f}"
    )
    print(
        f"[{name}_autotimes_base_vs_csv] "
        f"mse={base_metrics['mse']:.10f}, "
        f"mae={base_metrics['mae']:.10f}, "
        f"rmse={base_metrics['rmse']:.10f}"
    )

    return final_metrics, base_metrics, total_n


def evaluate_multiple_dirs_combined_against_csv(
    booster: xgb.Booster,
    parquet_dirs: List[str],
    feature_cols: List[str],
    gt_df: pd.DataFrame,
    station_map_df: pd.DataFrame,
    y_mu: np.ndarray,
    y_sd: np.ndarray,
    name: str = "all",
):
    final_sse = 0.0
    final_sae = 0.0

    base_sse = 0.0
    base_sae = 0.0

    total_n = 0

    for parquet_dir in parquet_dirs:
        files = sorted(glob.glob(os.path.join(parquet_dir, "part_*.parquet")))
        if not files:
            raise ValueError(f"No parquet parts found in {parquet_dir}")

        print(f"[EVAL-{name}] scanning dir: {parquet_dir}, parts={len(files)}")

        for i, f in enumerate(files, 1):
            print(f"[EVAL-{name}] reading {i}/{len(files)} from {os.path.basename(parquet_dir)}: {os.path.basename(f)}")
            df = pd.read_parquet(f)
            df = sanitize_parquet_df(df, feature_cols)
            merged = merge_with_ground_truth_no_shift(
                parquet_df=df,
                gt_df=gt_df,
                station_map_df=station_map_df,
            )
            if len(merged) == 0:
                continue

            print_alignment_check_scaled_and_raw(name, merged, y_mu, y_sd)

            sid_idx_arr = merged["sid_idx"].astype(int).to_numpy()

            y_true_raw_from_parquet = inverse_transform_by_sid(
                merged["y_true"].to_numpy(),
                sid_idx_arr,
                y_mu,
                y_sd
            )

            align_abs_diff = np.abs(y_true_raw_from_parquet - merged["y_true_csv"].to_numpy())
            keep_mask = align_abs_diff <= 1e-3

            print(
                f"[ALIGN FILTER-{name}] keep={keep_mask.sum()}/{len(keep_mask)}, "
                f"drop={(~keep_mask).sum()}, "
                f"keep_ratio={keep_mask.mean():.6f}"
            )

            merged = merged.loc[keep_mask].reset_index(drop=True)

            if len(merged) == 0:
                continue

            # 一定要在过滤后重新取
            X = merged[feature_cols]
            sid_idx_arr = merged["sid_idx"].astype(int).to_numpy()

            # parquet / xgb 输出在 scaled 空间
            y_hat_T_scaled = merged["y_hat_T"].to_numpy()
            pred_residual_scaled = booster.predict(xgb.DMatrix(X))
            pred_final_scaled = y_hat_T_scaled + pred_residual_scaled

            y_hat_T_raw = inverse_transform_by_sid(y_hat_T_scaled, sid_idx_arr, y_mu, y_sd)
            pred_final_raw = inverse_transform_by_sid(pred_final_scaled, sid_idx_arr, y_mu, y_sd)

            y_true_raw = merged["y_true_csv"].to_numpy()

            final_err = y_true_raw - pred_final_raw
            final_sse += float(np.sum(final_err ** 2))
            final_sae += float(np.sum(np.abs(final_err)))

            base_err = y_true_raw - y_hat_T_raw
            base_sse += float(np.sum(base_err ** 2))
            base_sae += float(np.sum(np.abs(base_err)))

            total_n += len(merged)

    if total_n == 0:
        raise RuntimeError(f"[EVAL-{name}] no matched rows between parquet and original csv")

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
        f"[{name}_final_vs_csv] "
        f"mse={final_metrics['mse']:.10f}, "
        f"mae={final_metrics['mae']:.10f}, "
        f"rmse={final_metrics['rmse']:.10f}"
    )
    print(
        f"[{name}_autotimes_base_vs_csv] "
        f"mse={base_metrics['mse']:.10f}, "
        f"mae={base_metrics['mae']:.10f}, "
        f"rmse={base_metrics['rmse']:.10f}"
    )

    return final_metrics, base_metrics, total_n


# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--raw_csv_path", type=str, required=True)
    parser.add_argument("--station_map_csv", type=str, required=True)

    parser.add_argument("--ms_train_ratio", type=float, default=0.7)
    parser.add_argument("--eval_max_parts", type=int, default=None,
                        help="不传则全量评估")


    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    probe_dir = args.train_dir or args.val_dir or args.test_dir
    if probe_dir is None:
        raise ValueError("At least one of train_dir / val_dir / test_dir must be provided")

    raw_feature_cols = get_feature_cols()
    cols = set(read_one_part_columns(probe_dir))
    feature_cols = [c for c in raw_feature_cols if c in cols]

    missing_core = [c for c in ["sid_idx", "horizon", "y_hat_T", "target_time"] if c not in cols]
    if missing_core:
        raise ValueError(f"Missing required parquet columns: {missing_core}")

    print("[INFO] feature cols =", feature_cols)

    booster = xgb.Booster()
    booster.load_model(args.model_path)
    print("[INFO] loaded model from", args.model_path)

    gt_df = load_ground_truth_csv(args.raw_csv_path)
    print("[INFO] loaded raw csv from", args.raw_csv_path)

    station_map_df = load_station_map_csv(args.station_map_csv)
    print("[INFO] loaded station map from", args.station_map_csv)

    scaler_info = build_scaler_from_raw_csv(
        raw_csv_path=args.raw_csv_path,
        train_ratio=args.ms_train_ratio,
    )
    y_mu = scaler_info["y_mu"]
    y_sd = scaler_info["y_sd"]

    print("[INFO] built scaler directly from raw csv")
    print("[INFO] y_mu shape:", y_mu.shape, "y_sd shape:", y_sd.shape)
    results = {
        "feature_cols": feature_cols,
        "model_path": args.model_path,
        "raw_csv_path": args.raw_csv_path,
        "station_map_csv": args.station_map_csv,
        "eval_max_parts": args.eval_max_parts,
    }

    if args.train_dir is not None:
        print("[STAGE] evaluating train against raw csv...")
        train_final, train_base, train_n = evaluate_parquet_dir_against_csv(
            booster=booster,
            parquet_dir=args.train_dir,
            feature_cols=feature_cols,
            gt_df=gt_df,
            station_map_df=station_map_df,
            y_mu=y_mu,
            y_sd=y_sd,
            name="train",
            max_parts=args.eval_max_parts,
        )
        results["train_final_vs_csv"] = train_final
        results["train_autotimes_base_vs_csv"] = train_base
        results["train_num_samples"] = train_n

    if args.val_dir is not None:
        print("[STAGE] evaluating val against raw csv...")
        val_final, val_base, val_n = evaluate_parquet_dir_against_csv(
            booster=booster,
            parquet_dir=args.val_dir,
            feature_cols=feature_cols,
            gt_df=gt_df,
            station_map_df=station_map_df,
            y_mu=y_mu,
            y_sd=y_sd,
            name="val",
            max_parts=args.eval_max_parts,
        )
        results["val_final_vs_csv"] = val_final
        results["val_autotimes_base_vs_csv"] = val_base
        results["val_num_samples"] = val_n

    if args.test_dir is not None:
        print("[STAGE] evaluating test against raw csv...")
        test_final, test_base, test_n = evaluate_parquet_dir_against_csv(
            booster=booster,
            parquet_dir=args.test_dir,
            feature_cols=feature_cols,
            gt_df=gt_df,
            station_map_df=station_map_df,
            y_mu=y_mu,
            y_sd=y_sd,
            name="test",
            max_parts=args.eval_max_parts,
        )
        results["test_final_vs_csv"] = test_final
        results["test_autotimes_base_vs_csv"] = test_base
        results["test_num_samples"] = test_n

    all_dirs = []
    if args.train_dir is not None:
        all_dirs.append(args.train_dir)
    if args.val_dir is not None:
        all_dirs.append(args.val_dir)
    if args.test_dir is not None:
        all_dirs.append(args.test_dir)

    print("[STAGE] evaluating combined train+val+test against raw csv...")
    all_final, all_base, all_n = evaluate_multiple_dirs_combined_against_csv(
        booster=booster,
        parquet_dirs=all_dirs,
        feature_cols=feature_cols,
        gt_df=gt_df,
        station_map_df=station_map_df,
        y_mu=y_mu,
        y_sd=y_sd,
        name="all",
    )

    results["all_final_vs_csv"] = all_final
    results["all_autotimes_base_vs_csv"] = all_base
    results["all_num_samples"] = all_n

    with open(os.path.join(args.out_dir, "eval_metrics_summary_vs_csv.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("[DONE] evaluation finished")

def debug_alignment_sample(
    parquet_dir: str,
    raw_csv_path: str,
    station_map_csv: str,
    train_ratio: float = 0.7,
    n_rows: int = 30,
    out_csv: Optional[str] = None,
):
    """
    调试 target_time / forecast_start_time / horizon 与原始 CSV datetime 的真实对应关系。
    会同时输出：
    1) parquet 原始 scaled 值
    2) inverse transform 后的 raw 值
    3) 多种候选时间下 raw csv 的 target

    候选时间包括：
    - target_time + shift
    - forecast_start_time + (horizon-1)*15min + shift
    - forecast_start_time + horizon*15min + shift
    """
    # 读取一个 parquet part
    files = sorted(glob.glob(os.path.join(parquet_dir, "part_*.parquet")))
    if not files:
        raise ValueError(f"No parquet parts found in {parquet_dir}")

    pq = pd.read_parquet(files[0]).head(n_rows).copy()

    needed_cols = ["sid_idx", "target_time", "horizon", "y_true", "y_hat_T"]
    missing = [c for c in needed_cols if c not in pq.columns]
    if missing:
        raise ValueError(f"parquet missing required columns for debug: {missing}")

    # forecast_start_time 有就用，没有也照样能跑
    has_forecast_start = "forecast_start_time" in pq.columns

    pq["target_time"] = pd.to_datetime(pq["target_time"])
    if has_forecast_start:
        pq["forecast_start_time"] = pd.to_datetime(pq["forecast_start_time"])

    # 读取原始 CSV
    raw = pd.read_csv(raw_csv_path)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw[["datetime", "station", "station_id", "target"]].copy()

    # 读取映射表
    smap = pd.read_csv(station_map_csv)[["station", "station_id"]].drop_duplicates()
    smap = smap.rename(columns={"station_id": "sid_idx", "station": "station_name_from_map"})

    # 挂站点名称
    pq = pq.merge(smap, on="sid_idx", how="left", validate="many_to_one")

    # 构造 scaler（复刻 Dataset_MultiStation_Custom 缩放）
    scaler_info = build_scaler_from_raw_csv(
        raw_csv_path=raw_csv_path,
        train_ratio=train_ratio,
    )
    y_mu = scaler_info["y_mu"]
    y_sd = scaler_info["y_sd"]

    rows = []
    shifts = [0, -15, -30, -45, -60, -75, -90, 15, 30, 45, 60]

    for _, r in pq.iterrows():
        sid = int(r["sid_idx"])
        tt = pd.Timestamp(r["target_time"])
        horizon = int(r["horizon"])

        station_name = r.get("station_name_from_map", None)

        y_true_scaled = float(r["y_true"])
        y_hat_scaled = float(r["y_hat_T"])

        y_true_raw = float(inverse_transform_by_sid(
            np.array([y_true_scaled], dtype=np.float32),
            np.array([sid], dtype=np.int64),
            y_mu,
            y_sd
        )[0])

        y_hat_raw = float(inverse_transform_by_sid(
            np.array([y_hat_scaled], dtype=np.float32),
            np.array([sid], dtype=np.int64),
            y_mu,
            y_sd
        )[0])

        sub = raw[raw["station_id"] == sid].copy()
        if len(sub) == 0:
            continue

        candidate = {
            "sid_idx": sid,
            "station_name": station_name,
            "forecast_start_time": pd.Timestamp(r["forecast_start_time"]) if has_forecast_start else pd.NaT,
            "target_time": tt,
            "horizon": horizon,
            "parquet_y_true_scaled": y_true_scaled,
            "parquet_y_hat_scaled": y_hat_scaled,
            "parquet_y_true_raw": y_true_raw,
            "parquet_y_hat_raw": y_hat_raw,
        }

        # 候选基准 1：直接用 target_time
        for shift in shifts:
            t1 = tt + pd.Timedelta(minutes=shift)
            hit = sub[sub["datetime"] == t1]
            candidate[f"csv_target__target_time_shift_{shift:+d}m"] = (
                float(hit["target"].iloc[0]) if len(hit) > 0 else np.nan
            )

        # 候选基准 2：forecast_start_time + (horizon-1)*15min
        if has_forecast_start:
            fst = pd.Timestamp(r["forecast_start_time"])
            base_a = fst + pd.Timedelta(minutes=(horizon - 1) * 15)
            candidate["candidate_base_a"] = base_a

            for shift in shifts:
                t2 = base_a + pd.Timedelta(minutes=shift)
                hit = sub[sub["datetime"] == t2]
                candidate[f"csv_target__fstart_plus_hm1_shift_{shift:+d}m"] = (
                    float(hit["target"].iloc[0]) if len(hit) > 0 else np.nan
                )

            # 候选基准 3：forecast_start_time + horizon*15min
            base_b = fst + pd.Timedelta(minutes=horizon * 15)
            candidate["candidate_base_b"] = base_b

            for shift in shifts:
                t3 = base_b + pd.Timedelta(minutes=shift)
                hit = sub[sub["datetime"] == t3]
                candidate[f"csv_target__fstart_plus_h_shift_{shift:+d}m"] = (
                    float(hit["target"].iloc[0]) if len(hit) > 0 else np.nan
                )

        rows.append(candidate)

    out = pd.DataFrame(rows)

    # 再自动算一组“哪个候选时间最接近 parquet_y_true_raw”
    compare_cols = [c for c in out.columns if c.startswith("csv_target__")]
    for c in compare_cols:
        out[f"absdiff__{c}"] = (out["parquet_y_true_raw"] - out[c]).abs()

    if out_csv is not None:
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[DEBUG] saved alignment sample to: {out_csv}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(out.head(n_rows).to_string(index=False))

    return out
if __name__ == "__main__":
    main()
    '''python /root/autotimes/eval_saved_xgb_model.py \
  --model_path /root/autodl-tmp/xgb_exports/CKPT_From_multi_scale_time-pt_no_prefix/no_exog/output/xgb_extmem.json \
  --train_dir /root/autodl-tmp/xgb_exports/CKPT_From_multi_scale_time-pt_no_prefix/no_exog/train_predictions \
  --val_dir /root/autodl-tmp/xgb_exports/CKPT_From_multi_scale_time-pt_no_prefix/no_exog/val_predictions \
  --test_dir /root/autodl-tmp/xgb_exports/CKPT_From_multi_scale_time-pt_no_prefix/no_exog/test_predictions \
  --out_dir /root/autodl-tmp/xgb_exports/CKPT_From_multi_scale_time-pt_no_prefix/no_exog/eval_vs_csv_dedup_latest \
  --raw_csv_path /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --station_map_csv /root/autodl-tmp/datasets/SelfMadeAusgridData/station_map.csv \
  --ms_train_ratio 0.7'''