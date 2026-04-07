import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
#from data_provider.m4 import M4Dataset, M4Meta
from sklearn.preprocessing import StandardScaler
from utils.tools import convert_tsf_to_dataframe
import warnings
import re
warnings.filterwarnings('ignore')

class Dataset_Preprocess(Dataset):
    """
    AutoTimes-style preprocess dataset (time-only):
    - Build prompts purely from a unique, sorted datetime axis.
    - Each time point -> one prompt: from t to t + (token_len-1)*freq
    """
    def __init__(self, root_path, size=None, data_path=None, freq_minutes=15):
        assert size is not None
        self.seq_len, self.label_len, self.pred_len = size
        self.token_len = self.seq_len - self.label_len
        self.freq_minutes = int(freq_minutes)

        fp = os.path.join(root_path, data_path) if root_path else data_path
        df = pd.read_csv(fp, usecols=["datetime"])  # ✅ 只读时间
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])

        # ✅ 多站点 long-format：取唯一时间轴
        dt = df["datetime"].drop_duplicates().sort_values().reset_index(drop=True)
        self.dt = dt.tolist()

    def __len__(self):
        return len(self.dt)

    def __getitem__(self, idx):
        start = self.dt[idx]
        end = start + datetime.timedelta(minutes=self.freq_minutes * (self.token_len - 1))
        return (
            f"This is the series from {start:%Y/%m/%d %H:%M:%S} "
            f"to {end:%Y/%m/%d %H:%M:%S}."
        )
class Dataset_MultiStation_Custom(Dataset):
    """
    原始数据是 long-format：每行是一条 (datetime, station, target, ...)。

    先构造全局唯一时间轴 dt（所有站点共有的、排序好的时间序列）。

    每个站点的数据都 reindex 到 dt，这样每个站点都有长度一样的序列（缺失会被填）。

    按全局时间轴切 train/val/test，避免同一时间点跨集合泄漏。

    time_only.pt 也是按这个全局时间轴生成的，所以切片索引必须用全局 time index。
    """
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        data_path="",
        time_pt_path="",
        weather_pt_path='',
        freq_minutes=15,
        scale=False,
        train_ratio=0.7,
        val_ratio=0.1,
        require_contiguous=False,
        fillna_value=None,
        return_sid=True,
        exog_cols=None,
        token_len=None,
        weather_read_mode="torch",
        weather_shard_dir="",
        holiday_csv_path=None,
        use_prefix=False,
        use_social_prefix=False,
        social_csv_path=None,
    ):

        assert size is not None, "size must be provided"
        assert flag in ["train", "val", "test"]
        self.holiday_csv_path = holiday_csv_path
        self.use_prefix = use_prefix
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.seq_len, self.label_len, self.pred_len = size
        self.use_social_prefix = use_social_prefix
        self.social_csv_path = social_csv_path
        if token_len is None:
            self.token_len = self.seq_len - self.label_len
        else:
            self.token_len = int(token_len)

        self.freq_minutes = int(freq_minutes)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.require_contiguous = bool(require_contiguous)
        self.fillna_value = fillna_value
        self.return_sid = bool(return_sid)
        self.scale = bool(scale)
        self.exog_cols = exog_cols or []

        self.use_time = bool(time_pt_path)
        self.weather_read_mode = weather_read_mode
        self.weather_shard_dir = weather_shard_dir

        self.weather_pt = None
        self.weather_shards = []
        self._loaded_shard = None
        self._loaded_shard_path = None

        self.use_weather = (
            (self.weather_read_mode == "torch" and bool(weather_pt_path)) or
            (self.weather_read_mode == "shard" and bool(weather_shard_dir))
        )

        if not self.use_time and not self.use_weather:
            raise ValueError("At least one of time_pt_path or weather input must be set")

        if self.use_time:
            print("[LOAD] time_pt_path =", time_pt_path)
            self.time_pt = torch.load(time_pt_path, map_location="cpu")
            print("[TIME_PT] shape:", self.time_pt.shape, "dtype:", self.time_pt.dtype)
            print("[TIME_PT] sig sum0:", float(self.time_pt[0].float().sum()))
            print("[TIME_PT] sig mean:", float(self.time_pt.float().mean()))
        else:
            self.time_pt = None
            print("[TIME_PT] not provided")

        self.weather_pt_path = weather_pt_path

        # 先读 CSV，得到 T / border1 / border2 / stations / Y
        self.__read_data__()

        if self.use_weather:
            if self.weather_read_mode == "torch":
                print("[LOAD] weather_pt_path =", self.weather_pt_path)
                full_weather = torch.load(self.weather_pt_path, map_location="cpu")
                print("[WEATHER_PT FULL] shape:", full_weather.shape, "dtype:", full_weather.dtype)

                if full_weather.shape[0] != len(self.stations):
                    raise ValueError(
                        f"weather station mismatch: weather={full_weather.shape[0]} vs stations={len(self.stations)}"
                    )
                if full_weather.shape[1] != self.T:
                    raise ValueError(
                        f"weather time mismatch: weather={full_weather.shape[1]} vs T={self.T}"
                    )

                self.weather_pt = full_weather[:, self.border1:self.border2, :].contiguous()
                del full_weather
                print("[WEATHER_PT SPLIT] loaded shape:", self.weather_pt.shape)

            elif self.weather_read_mode == "shard":
                print("[LOAD] weather_shard_dir =", self.weather_shard_dir)

                for fname in os.listdir(self.weather_shard_dir):
                    m = re.match(r"merge_station_(\d+)_(\d+)\.pt$", fname)
                    if m:
                        start_sid = int(m.group(1))
                        end_sid = int(m.group(2))
                        self.weather_shards.append({
                            "path": os.path.join(self.weather_shard_dir, fname),
                            "start_sid": start_sid,
                            "end_sid": end_sid,
                        })

                self.weather_shards = sorted(self.weather_shards, key=lambda x: x["start_sid"])

                if len(self.weather_shards) == 0:
                    raise ValueError(f"No valid shard files found in {self.weather_shard_dir}")

                print("[WEATHER_SHARD] found shards:")
                for s in self.weather_shards:
                    print("   ", s)

                covered = set()
                for s in self.weather_shards:
                    covered.update(range(s["start_sid"], s["end_sid"] + 1))

                expected = set(range(len(self.stations)))
                if covered != expected:
                    missing = sorted(expected - covered)
                    extra = sorted(covered - expected)
                    raise ValueError(
                        f"weather shard coverage mismatch. missing={missing[:10]}, extra={extra[:10]}"
                    )
            else:
                raise ValueError(f"Unknown weather_read_mode: {self.weather_read_mode}")
        else:
            self.weather_pt = None
            self.weather_shards = []
            print("[WEATHER_PT] not provided, use time_pt only")
        self.social_cols = [
            "pop_density_per_sqkm_norm",
            "young_ratio_0_14_norm",
            "working_ratio_15_64_norm",
            "elderly_ratio_65_plus_norm",
            "private_dwelling_ratio_norm",
            "other_dwelling_ratio_norm",
        ]

        self.social_dict = {}

        if self.use_social_prefix and self.social_csv_path is not None and os.path.exists(self.social_csv_path):
            df_s = pd.read_csv(self.social_csv_path, skipinitialspace=True)
            df_s.columns = df_s.columns.str.strip()

            df_s["stationname"] = df_s["stationname"].astype(str).str.strip()

            for col in self.social_cols:
                df_s[col] = pd.to_numeric(df_s[col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

            for _, row in df_s.iterrows():
                station_name = row["stationname"]
                vec = row[self.social_cols].values.astype(np.float32)
                self.social_dict[station_name] = vec
            self.sid_to_station = {}
            if self.use_social_prefix:
                for sid_idx, station_name in enumerate(self.stations):
                    self.sid_to_station[sid_idx] = str(station_name).strip()
        self.__build_index_map__()
    def _build_social_prefix(self, sid_idx):
        if not self.use_social_prefix:
            return None

        station_name = self.sid_to_station.get(sid_idx, None)
        if station_name is None or station_name not in self.social_dict:
            return torch.zeros(len(self.social_cols), dtype=torch.float32)

        return torch.tensor(self.social_dict[station_name], dtype=torch.float32)
    def _get_weather_station_tensor_from_shard(self, sid_idx):
        for shard in self.weather_shards:
            if shard["start_sid"] <= sid_idx <= shard["end_sid"]:
                local_sid = sid_idx - shard["start_sid"]

                if self._loaded_shard_path != shard["path"]:
                    self._loaded_shard = torch.load(shard["path"], map_location="cpu")
                    self._loaded_shard_path = shard["path"]

                    if self._loaded_shard.ndim != 3:
                        raise ValueError(
                            f"Shard {shard['path']} must be [N,T,D], got {tuple(self._loaded_shard.shape)}"
                        )
                    if self._loaded_shard.shape[1] != self.T:
                        raise ValueError(
                            f"Shard {shard['path']} time dim mismatch: "
                            f"{self._loaded_shard.shape[1]} vs T={self.T}"
                        )

                if not (0 <= local_sid < self._loaded_shard.shape[0]):
                    raise IndexError(
                        f"local_sid={local_sid} out of range for shard {shard['path']} "
                        f"with shape {tuple(self._loaded_shard.shape)}"
                    )

                return self._loaded_shard[local_sid]

        raise RuntimeError(f"sid_idx={sid_idx} not in any shard")
    def __read_data__(self):#把 CSV long 表 → 对齐成 [N_station, T, ...]
        fp = os.path.join(self.root_path, self.data_path)
        usecols = ["datetime", "station", "target"] + list(self.exog_cols)
        df = pd.read_csv(fp, usecols=usecols)

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime", "station", "target"])
        df["station"] = df["station"].astype(str)

        # 全局时间轴：unique + sort
        self.dt = (
            df["datetime"]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )
        self.T = len(self.dt)

        # 校验 time_only.pt 长度一致
        # time_only.pt 的第 t 行 embedding 对应 dt 的第 t 个时间点。
        if self.time_pt is not None:
            if self.time_pt.shape[0] != self.T:
                raise ValueError(
                    f"time_only.pt length mismatch: time_pt={self.time_pt.shape[0]} vs dt={self.T}. "
                    "请确认 time_only.pt 是用同一份 merged_clean.csv 的 unique datetime 生成的。"
                )

        # station 列表
        self.stations = sorted(df["station"].unique().tolist())
        self.sid2idx = {s: i for i, s in enumerate(self.stations)}
        
        # 把每个 station reindex 到全局时间轴
        # 目标：self.Y shape [N_station, T, 1]
        N = len(self.stations)
        Y = np.full((N, self.T), np.nan, dtype=np.float32)

        # （可选）外生变量：self.X_exog shape [N_station, T, exog_dim]
        exog_dim = len(self.exog_cols)
        X_exog = None
        if exog_dim > 0:
            X_exog = np.full((N, self.T, exog_dim), np.nan, dtype=np.float32)

        # 用索引加速 reindex
        dt_index = pd.Index(self.dt)

        for s in self.stations:
            sid_idx = self.sid2idx[s]
            sdf = df[df["station"] == s].sort_values("datetime")

            # 如果同一 station 同一 datetime 有重复，先聚合（均值/最后一条都行，这里取均值）
            sdf = sdf.groupby("datetime", as_index=True).mean(numeric_only=True)

            # reindex 到全局 dt
            sdf = sdf.reindex(dt_index)
            # 对 target 插值补全
            s = sdf["target"].astype("float32")
            s = s.interpolate(limit_direction="both")
            s = s.ffill().bfill()
            sdf["target"] = s
            Y[sid_idx, :] = sdf["target"].to_numpy(dtype=np.float32)
            #这里假设外生变量也是按站点随时间变化的（比如气温），如果是站点静态属性（人口密度），时间填充成常数序列是 OK 的。
            if exog_dim > 0:
                for j, c in enumerate(self.exog_cols):
                    ss = sdf[c].astype("float32")
                    ss = ss.interpolate(limit_direction="both")
                    ss = ss.ffill().bfill()
                    sdf[c] = ss
                    X_exog[sid_idx, :, j] = ss.to_numpy(dtype=np.float32)


        # 若指定 fillna_value，把剩余 NaN 填成固定值
        if self.fillna_value is not None:
            Y = np.nan_to_num(Y, nan=float(self.fillna_value))
            if X_exog is not None:
                X_exog = np.nan_to_num(X_exog, nan=float(self.fillna_value))

        self.Y = Y[:, :, None]  # [N, T, 1]
        
        # 先根据整体时间轴计算 split，用于后续归一化时使用训练区间
        num_train = int(self.T * self.train_ratio)
        num_val = int(self.T * self.val_ratio)
        num_test = self.T - num_train - num_val

        if self.scale:
            Y2 = self.Y[..., 0]  # [N, T]
            train_slice = slice(0, num_train)

            mu = np.mean(Y2[:, train_slice], axis=1, keepdims=True)
            sd = np.std(Y2[:, train_slice], axis=1, keepdims=True)
            sd = np.maximum(sd, 1e-6)

            Y2 = (Y2 - mu) / sd
            self.Y = Y2[:, :, None].astype(np.float32)

            # 记录下来，后续如果你要反归一化/画图会用到
            self.y_mu = mu.astype(np.float32)
            self.y_sd = sd.astype(np.float32)
        self.X_exog = X_exog    # [N, T, exog_dim] or None  目前只是存着，__getitem__ 里没返回


        border1s = [0, num_train, num_train + num_val]
        border2s = [num_train, num_train + num_val, self.T]

        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[self.flag]
        self.border1 = border1s[self.set_type]
        self.border2 = border2s[self.set_type]

        # contiguous 检测（全局时间轴）
        if self.require_contiguous:
            # breaks[t]=1 表示 dt[t]-dt[t-1] != freq
            diffs = self.dt.diff().dt.total_seconds().to_numpy()
            step = self.freq_minutes * 60
            breaks = np.zeros(self.T, dtype=np.int32)
            breaks[1:] = (diffs[1:] != step).astype(np.int32)
            # prefix sum 方便 O(1) 判断窗口内是否有断点
            self.break_prefix = np.cumsum(breaks)  # break_prefix[t] = breaks[:t+1] sum
        else:
            self.break_prefix = None

        #print(f"[WEATHER] weather_pt loaded: {self.weather_pt is not None}", flush=True)
        if self.weather_pt is not None:
            print(f"[WEATHER] weather_pt shape={tuple(self.weather_pt.shape)} dtype={self.weather_pt.dtype}", flush=True)
            print(f"[WEATHER] N(stations)={len(self.stations)} T(dt)={self.T}", flush=True)
        #holiday prefix
        self.holiday_dates = set()
        if self.holiday_csv_path is not None and os.path.exists(self.holiday_csv_path):
            df_h = pd.read_csv(self.holiday_csv_path)
            df_h["date"] = pd.to_datetime(df_h["date"]).dt.date
            df_h["is_holiday"] = df_h["is_holiday"].astype(str).str.lower().map({
                "true": True,
                "false": False,
                "1": True,
                "0": False
            })

            self.holiday_dates = set(df_h.loc[df_h["is_holiday"] == True, "date"].tolist())
    def __window_has_break(self, l, r):
        """
        判断 (l, r] 区间内是否存在 breaks（即 l+1..r 有断点）
        我们希望窗口 [l, r] 内连续 => l+1..r 都不能断
        """
        if self.break_prefix is None:
            return False
        if r <= l:
            return False
        # breaks in [l+1..r] -> prefix[r] - prefix[l]
        return (self.break_prefix[r] - self.break_prefix[l]) > 0

    def __build_index_map__(self):
        """
        index_map: list of (sid_idx, s_begin_global)
        约束：s_begin + seq_len + pred_len <= border2
        且 s_begin >= border1
        """
        self.index_map = []
        max_begin = self.border2 - (self.seq_len + self.pred_len)
        if max_begin < self.border1:
            # 该 split 太短，没法采样
            return

        for sid_idx in range(len(self.stations)):
            for s_begin in range(self.border1, max_begin + 1):
                # 可选：要求输入+预测窗口在全局时间轴连续
                if self.require_contiguous:
                    w_end = s_begin + self.seq_len + self.pred_len - 1
                    if self.__window_has_break(s_begin, w_end):
                        continue
                self.index_map.append((sid_idx, s_begin))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        sid_idx, s_begin = self.index_map[i]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.Y[sid_idx, s_begin:s_end, :]
        seq_y = self.Y[sid_idx, r_begin:r_end, :]

        # ===== 1. 先构建 time =====
        if self.time_pt is not None:
            seq_x_mark = self.time_pt[s_begin:s_end:self.token_len]
            seq_y_mark = self.time_pt[s_end:r_end:self.token_len]
        else:
            seq_x_mark = None
            seq_y_mark = None

        # ===== 2. weather =====
        if self.use_weather:
            if self.weather_read_mode == "torch":
                local_s_begin = s_begin - self.border1
                local_s_end = s_end - self.border1
                local_r_end = r_end - self.border1

                w_x_mark = self.weather_pt[sid_idx, local_s_begin:local_s_end:self.token_len, :]
                w_y_mark = self.weather_pt[sid_idx, local_s_end:local_r_end:self.token_len, :]

            elif self.weather_read_mode == "shard":
                station_weather = self._get_weather_station_tensor_from_shard(sid_idx)   # [T, D]
                w_x_mark = station_weather[s_begin:s_end:self.token_len, :]
                w_y_mark = station_weather[s_end:r_end:self.token_len, :]

            else:
                raise ValueError(f"Unknown weather_read_mode: {self.weather_read_mode}")

            if seq_x_mark is None:
                seq_x_mark = w_x_mark
                seq_y_mark = w_y_mark
            else:
                seq_x_mark = torch.cat([seq_x_mark, w_x_mark], dim=-1)
                seq_y_mark = torch.cat([seq_y_mark, w_y_mark], dim=-1)

        # ===== 3. safety =====
        if seq_x_mark is None:
            raise ValueError("Both time_pt and weather_pt are None!")
        # -------------------------
        # 4) safety checks
        # -------------------------
        expected_x_tokens = self.seq_len // self.token_len
        if seq_x_mark.shape[0] != expected_x_tokens:
            raise RuntimeError(
                f"x_mark token num mismatch: got {seq_x_mark.shape[0]}, "
                f"expected {expected_x_tokens}"
            )

        expected_y_tokens = self.pred_len // self.token_len
        if self.pred_len % self.token_len != 0:
            expected_y_tokens += 1

        if seq_y_mark.shape[0] != expected_y_tokens:
            raise RuntimeError(
                f"y_mark token num mismatch: got {seq_y_mark.shape[0]}, "
                f"expected {expected_y_tokens}"
            )

        assert seq_x_mark.std() > 0, "time.pt loaded but seq_x_mark is zero!"
        if not hasattr(self, "_dbg_markdim_once"):
            self._dbg_markdim_once = True
            print("[MARK DIM]", seq_x_mark.shape[-1], flush=True)

        prefix_calendar = None
        if self.use_prefix:
            # 预测第一个未来 token 的起始时间
            ts_prefix = pd.Timestamp(self.dt[s_end])
            prefix_calendar = self._build_calendar_prefix(ts_prefix)

        prefix_social = None
        if self.use_social_prefix:
            prefix_social = self._build_social_prefix(sid_idx)

        if prefix_calendar is None:
            prefix_calendar = torch.zeros(18, dtype=torch.float32)

        if prefix_social is None:
            prefix_social = torch.zeros(6, dtype=torch.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, prefix_calendar, prefix_social

    def _build_calendar_prefix(self, ts):
        """
        ts: pandas.Timestamp
        return: torch.FloatTensor [18]
        feature order:
        [month_norm] +
        [season_onehot(4)] +
        [dow_onehot(7)] +
        [day_type_onehot(3: workday/weekend/holiday)] +
        [holiday_rel_onehot(3: holiday-1 / holiday / holiday+1)]
        """

        # 1) month_norm
        month_norm = [ts.month / 12.0]

        # 2) season onehot (Australian / NSW seasons)
        # spring: 9,10,11
        # summer: 12,1,2
        # autumn: 3,4,5
        # winter: 6,7,8
        # 顺序固定为 [spring, summer, autumn, winter]
        season = [0.0, 0.0, 0.0, 0.0]
        if ts.month in [9, 10, 11]:
            season[0] = 1.0   # spring
        elif ts.month in [12, 1, 2]:
            season[1] = 1.0   # summer
        elif ts.month in [3, 4, 5]:
            season[2] = 1.0   # autumn
        else:
            season[3] = 1.0   # winter

        # 3) day-of-week onehot: Monday=0 ... Sunday=6
        dow = [0.0] * 7
        dow[ts.weekday()] = 1.0

        # 4) holiday / weekend / workday
        d = ts.date()
        is_holiday = d in self.holiday_dates
        is_weekend = ts.weekday() >= 5
        is_workday = (not is_weekend) and (not is_holiday)

        day_type = [0.0, 0.0, 0.0]  # [workday, weekend, holiday]
        if is_workday:
            day_type[0] = 1.0
        elif is_weekend and (not is_holiday):
            day_type[1] = 1.0
        else:
            day_type[2] = 1.0

        # 5) holiday relative: holiday-1 / holiday / holiday+1
        prev_day = (ts - pd.Timedelta(days=1)).date()
        next_day = (ts + pd.Timedelta(days=1)).date()

        holiday_rel = [0.0, 0.0, 0.0]
        if next_day in self.holiday_dates:
            holiday_rel[0] = 1.0   # holiday-1：明天是节假日
        if d in self.holiday_dates:
            holiday_rel[1] = 1.0   # holiday：今天是节假日
        if prev_day in self.holiday_dates:
            holiday_rel[2] = 1.0   # holiday+1：昨天是节假日

        feat = month_norm + season + dow + day_type + holiday_rel
        return torch.tensor(feat, dtype=torch.float32)