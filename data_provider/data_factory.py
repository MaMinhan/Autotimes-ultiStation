from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data_provider.data_loader import Dataset_MultiStation_Custom
from data_provider.dataset_npy import Dataset_NPY



def data_provider(args, flag):

    if args.data == "npy":
        dataset = Dataset_NPY(
            x_path=args.npy_x_path,
            y_path=args.npy_y_path,
            sid_path=args.npy_sid_path,
            flag=flag,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.test_pred_len if flag == "test" else args.token_len,  # 这里见下方说明
            token_len=args.token_len,
            split_ratio=(args.npy_train_ratio, args.npy_val_ratio, 1.0 - args.npy_train_ratio - args.npy_val_ratio),
            mark_dim=args.npy_mark_dim,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(flag == "train"),
            num_workers=args.num_workers,
            drop_last=args.drop_last,
        )
        return dataset, loader
    if args.data == "custom_ms":
        pred_len = args.test_pred_len
        if flag == "train":
            print(f"[DATA_FACTORY] flag={flag}, seq_len={args.seq_len}, label_len={args.label_len}, "
                f"token_len={args.token_len}, pred_len(test_pred_len)={pred_len}")
        elif flag == "val":
            print(f"[DATA_FACTORY] flag={flag}, seq_len={args.seq_len}, label_len={args.label_len}, "
                f"token_len={args.token_len}, pred_len(test_pred_len)={pred_len}")
        elif flag == "test":
            print(f"[DATA_FACTORY] flag={flag}, seq_len={args.seq_len}, label_len={args.label_len}, "
                f"token_len={args.token_len}, pred_len(test_pred_len)={pred_len}")
        dataset = Dataset_MultiStation_Custom(
            root_path=args.root_path,
            flag=flag,
            size=[args.seq_len, args.label_len, pred_len],
            token_len=args.token_len,
            data_path=args.data_path,
            weather_pt_path=args.weather_pt_path,
            time_pt_path=args.time_pt_path,  
            freq_minutes=getattr(args, "freq_minutes", 15),
            
            # split（可先用默认）
            train_ratio=getattr(args, "ms_train_ratio", 0.7),
            val_ratio=getattr(args, "ms_val_ratio", 0.1),

            require_contiguous=getattr(args, "ms_require_contiguous", False),
            fillna_value=getattr(args, "ms_fillna_value", None),
            scale=getattr(args, "ms_scale", False),
            return_sid=False,  # 先别返回 sid，避免训练 loop 解包问题
        )
        
        sampler = None
        shuffle = (flag == "train")
        if getattr(args, "use_multi_gpu", False):
            # run.py 里 use_multi_gpu 会 init_process_group
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
        )
        return dataset, loader


    else:
        raise ValueError(f"Unknown args.data = {args.data}")
        
    