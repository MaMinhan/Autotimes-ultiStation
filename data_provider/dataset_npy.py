# data_provider/dataset_npy.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class Dataset_NPY(Dataset):
    """
    Load pre-cut window samples from X.npy / Y.npy / sid.npy (memmap).
    Returns:
      batch_x: [seq_len, 1]
      batch_y: [label_len + pred_len, 1]  (concat last label_len of x + y)
      batch_x_mark: [seq_len//token_len, 768]  (dummy zeros for now)
      batch_y_mark: [(label_len+pred_len)//token_len, 768] (dummy zeros for now)
    """
    def __init__(
        self,
        x_path,
        y_path,
        sid_path,
        flag="train",
        seq_len=672,
        label_len=576,
        pred_len=96,
        token_len=96,
        split_ratio=(0.7, 0.1, 0.2),
        mark_dim=768,
    ):
        assert flag in ["train", "val", "test"]
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.token_len = token_len
        self.mark_dim = mark_dim

        self.X = np.load(x_path, mmap_mode="r")   # [N, 672]
        self.Y = np.load(y_path, mmap_mode="r")   # [N, 96]
        self.S = np.load(sid_path, mmap_mode="r") # [N]
        assert len(self.X) == len(self.Y) == len(self.S)

        N = len(self.X)
        n_train = int(N * split_ratio[0])
        n_val   = int(N * split_ratio[1])
        # test = rest
        if flag == "train":
            self.start, self.end = 0, n_train
        elif flag == "val":
            self.start, self.end = n_train, n_train + n_val
        else:
            self.start, self.end = n_train + n_val, N

        # 预先算好 mark 的长度（AutoTimes 的 dataset 里通常是按 token_len 抽样）
        self.x_mark_len = self.seq_len // self.token_len
        self.y_mark_len = (self.label_len + self.pred_len) // self.token_len

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        i = self.start + idx
        x = self.X[i].astype(np.float32)  # [672]
        y_future = self.Y[i].astype(np.float32)  # [96]
        # 拼出 y: [label_len + pred_len] = 576 + 96 = 672
        y = np.concatenate([x[-self.label_len:], y_future], axis=0).astype(np.float32)

        # AutoTimes 原始接口是 [L, C]，这里 C=1
        seq_x = torch.from_numpy(x).unsqueeze(-1)  # [672, 1]
        seq_y = torch.from_numpy(y).unsqueeze(-1)  # [672, 1]

        # 先给一个“全 0 的 mark”，保证模型能跑通
        # 后面你要加入天气等外生变量，再把 mark 换成真实特征/embedding
        x_mark = torch.zeros((self.x_mark_len, self.mark_dim), dtype=torch.float32)
        y_mark = torch.zeros((self.y_mark_len, self.mark_dim), dtype=torch.float32)

        return seq_x, seq_y, x_mark, y_mark
