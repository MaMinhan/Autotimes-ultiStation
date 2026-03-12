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

warnings.filterwarnings('ignore')
'''

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True, seasonal_patterns=None, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data_name = self.data_path.split('.')[0]
        self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}.pt'))
        self.data_stamp = self.data_stamp[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True, seasonal_patterns=None, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
            

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        data_name = self.data_path.split('.')[0]
        self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}.pt'))
        self.data_stamp = self.data_stamp[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 seasonal_patterns=None, scale=True, drop_short=False):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None, data_path='ETTh1.csv',
                 scale=False, inverse=False, seasonal_patterns='Yearly', drop_short=False):
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class Dataset_TSF(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path=None,
                 scale=True, seasonal_patterns=None, drop_short=False):
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.pred_len
        self.context_len = 4 * self.token_len
        print(self.seq_len, self.label_len, self.pred_len)
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.drop_short = drop_short
        self.timeseries = self.__read_data__()


    def __read_data__(self):
        df, _, _, _, _ = convert_tsf_to_dataframe(os.path.join(self.root_path, self.data_path))
        def dropna(x):
            return x[~np.isnan(x)]
        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
        if self.drop_short:
            timeseries = [ts for ts in timeseries if ts.shape[0] > self.context_len]
        self.tot_len = 0
        self.len_seq = []
        self.seq_id = []
        for i in range(len(timeseries)):
            res_len = max(self.pred_len + self.seq_len - timeseries[i].shape[0], 0)
            pad_zeros = np.zeros(res_len)
            timeseries[i] = np.hstack([pad_zeros, timeseries[i]])

            _len = timeseries[i].shape[0]
            train_len = _len-self.pred_len
            border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]
            border2s = [train_len - self.pred_len,  train_len,                                _len]
            
            curr_len = border2s[self.set_type] - max(border1s[self.set_type], 0) - self.pred_len - self.seq_len + 1
            curr_len = max(0, curr_len)
            
            self.len_seq.append(np.zeros(curr_len) + self.tot_len)
            self.seq_id.append(np.zeros(curr_len) + i)
            self.tot_len += curr_len
            
        self.len_seq = np.hstack(self.len_seq)
        self.seq_id = np.hstack(self.seq_id)

        return timeseries

    def __getitem__(self, index):
        len_seq = self.len_seq[index]
        seq_id = int(self.seq_id[index])
        index = index - int(len_seq)

        _len = self.timeseries[seq_id].shape[0]
        train_len = _len - self.pred_len
        border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]

        s_begin = index + border1s[self.set_type]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        data_x = self.timeseries[seq_id][s_begin:s_end]
        data_y = self.timeseries[seq_id][r_begin:r_end]
        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)

        return data_x, data_y, data_x, data_y

    def __len__(self):
        return self.tot_len

class Dataset_TSF_ICL(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path=None,
                 scale=True, seasonal_patterns=None, drop_short=True):
        
        self.pred_len = size[2]
        self.token_len = self.pred_len
        self.context_len = 4 * self.token_len

        self.root_path = root_path
        self.data_path = data_path
        self.timeseries = self.__read_data__()

    def __read_data__(self):
        df, _, _, _, _ = convert_tsf_to_dataframe(os.path.join(self.root_path, self.data_path))
        def dropna(x):
            return x[~np.isnan(x)]
        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
        timeseries = [ts for ts in timeseries if ts.shape[0] > self.context_len]
        return timeseries

    # we uniformly adopting the first time points of the time series as the corresponding prompt.
    def __getitem__(self, index):        
        data_x1 = self.timeseries[index][:2*self.token_len]
        data_x2 = self.timeseries[index][-2*self.token_len:-1*self.token_len]
        data_x = np.concatenate((data_x1, data_x2))
        data_y = self.timeseries[index][-1*self.token_len:]
        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)
        return data_x, data_y, data_x, data_y

    def __len__(self):
        return len(self.timeseries)
'''
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
            f"This is Time Series from {start:%Y-%m-%d %H:%M:%S} "
            f"to {end:%Y-%m-%d %H:%M:%S}."
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
        size=None,                 # [seq_len, label_len, pred_len_or_token_len]
        data_path="/root/autodl-tmp/datasets/SelfMadeAusgridData/merged_include_id_filled.csv",
        time_pt_path="/root/autodl-tmp/datasets/SelfMadeAusgridData/merged_include_id_filled_by_GPT2.pt",         # e.g. /root/.../time_only_20260129.pt
        weather_pt_path='/root/autodl-tmp/datasets/SelfMadeAusgridData/weather_numbers_by_GPT2.pt',
        freq_minutes=15,
        scale=False,               # 先默认不做 scaler（multi-station + missing 更麻烦）
        train_ratio=0.7,
        val_ratio=0.1,
        require_contiguous=False,  # True: 窗口内必须严格 15min 连续，否则跳过
        fillna_value=None,         # None: keep NaN；或填 0 / 前向填充等（按你策略）
        return_sid=True,          # 如果需要站点级评估：用 sid_idx（return_sid=True）
        exog_cols=None,            # list[str] 外生变量列名（可选）
        token_len=None
    ):

        assert size is not None, "size must be provided"
        assert flag in ["train", "val", "test"]

        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        """pred_len 决定 seq_y 的未来长度（label+pred）
        token_len 决定 seq_x_mark/seq_y_mark 的采样步长
        token_len 并不等于 pred_len，而是固定等于 seq_len - label_len"""
        self.seq_len, self.label_len, self.pred_len = size
        #print("[DEBUG] seq_len,label_len,pred_len =", self.seq_len, self.label_len, self.pred_len)
        # AutoTimes 里 token_len 通常 = pred_len(训练时) 或你传入的 token_len
        # 这里为了和你 Step B 描述一致，token_len 用 args.token_len（训练时 size[2] 就是 token_len）
        if token_len is None:
            # 兜底：如果没传，就用原先那套（保持兼容）
            self.token_len = self.seq_len - self.label_len
        else:
            self.token_len = int(token_len)
        #print("[DEBUG] token_len(mark stride)     =", self.token_len)
        self.freq_minutes = int(freq_minutes)

        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.require_contiguous = bool(require_contiguous)
        self.fillna_value = fillna_value
        self.return_sid = bool(return_sid)

        # whether to apply per-station scaling (mean/std) on the target
        self.scale = bool(scale)

        self.exog_cols = exog_cols or []

        # 期望 shape [T_unique, d]
        if time_pt_path is None:
            raise ValueError("time_pt_path must be set to your shared time_only.pt")
        self.time_pt = torch.load(time_pt_path, map_location="cpu")  # [T, d]
        self.weather_pt = torch.load(weather_pt_path, map_location="cpu")
        self.__read_data__()
        self.__build_index_map__()

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

        # 支持 [T, D] 或 [N, T, D]
        if self.time_pt.dim() == 2:
            if self.time_pt.shape[0] != self.T:
                raise ValueError(
                    f"time.pt length mismatch: time_pt={self.time_pt.shape[0]} vs dt={self.T}"
                )
        elif self.time_pt.dim() == 3:
            # 约定格式 [N, T, D]
            if self.time_pt.shape[1] != self.T:
                raise ValueError(
                    f"time.pt length mismatch: time_pt.shape[1]={self.time_pt.shape[1]} vs dt={self.T}"
                )
        else:
            raise ValueError(f"Unsupported time.pt dim = {self.time_pt.dim()}, expected 2 or 3")
        print("[TIME_PT] loaded shape:", self.time_pt.shape)
        print("[TIME_PT] dim:", self.time_pt.dim())
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

        print(f"[WEATHER] weather_pt loaded: {self.weather_pt is not None}", flush=True)
        if self.weather_pt is not None:
            print(f"[WEATHER] weather_pt shape={tuple(self.weather_pt.shape)} dtype={self.weather_pt.dtype}", flush=True)
            print(f"[WEATHER] N(stations)={len(self.stations)} T(dt)={self.T}", flush=True)

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

        # target
        seq_x = self.Y[sid_idx, s_begin:s_end, :]   # [seq_len, 1]
        seq_y = self.Y[sid_idx, r_begin:r_end, :]   # [label_len+pred_len, 1]

        # time marks
        if self.time_pt.dim() == 2:
            # 老格式: [T, D]
            seq_x_mark = self.time_pt[s_begin:s_end:self.token_len]   # [token_num, D]
            seq_y_mark = self.time_pt[s_end:r_end:self.token_len]     # [token_num2, D]
        elif self.time_pt.dim() == 3:
            # 新格式: [N, T, D]
            seq_x_mark = self.time_pt[sid_idx, s_begin:s_end:self.token_len, :]   # [token_num, D]
            seq_y_mark = self.time_pt[sid_idx, s_end:r_end:self.token_len, :]     # [token_num2, D]
        else:
            raise ValueError(f"Unsupported time.pt dim = {self.time_pt.dim()}")

        # weather marks
        w_x_mark = self.weather_pt[sid_idx, s_begin:s_end:self.token_len, :]
        w_y_mark = self.weather_pt[sid_idx, s_end:r_end:self.token_len, :]

        # concat time + weather
        seq_x_mark = torch.cat([seq_x_mark, w_x_mark], dim=-1)
        seq_y_mark = torch.cat([seq_y_mark, w_y_mark], dim=-1)

        # # debug print：一定要放到定义之后
        # if i == 0:
        #     print("[GETITEM] sid_idx =", sid_idx)
        #     print("[GETITEM] seq_x.shape =", seq_x.shape)
        #     print("[GETITEM] seq_y.shape =", seq_y.shape)
        #     print("[GETITEM] seq_x_mark.shape =", seq_x_mark.shape)
        #     print("[GETITEM] seq_y_mark.shape =", seq_y_mark.shape)
        #     print("[GETITEM] w_x_mark.shape =", w_x_mark.shape)
        #     print("[GETITEM] w_y_mark.shape =", w_y_mark.shape)

        if self.return_sid:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, sid_idx

        assert seq_x_mark.std() > 0, "time.pt loaded but seq_x_mark is zero!"

        if not hasattr(self, "_dbg_markdim_once"):
            self._dbg_markdim_once = True
            print("[MARK DIM]", seq_x_mark.shape[-1], flush=True)

        return seq_x, seq_y, seq_x_mark, seq_y_mark