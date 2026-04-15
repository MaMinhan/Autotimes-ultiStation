from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings('ignore')

import pandas as pd

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        if self.args.use_multi_gpu:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        else:
            self.device = self.args.gpu
            model = model.to(self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print(n, p.dtype, p.shape)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    def _rollout_predict(self, batch_x, batch_x_mark, batch_y_mark, pred_len, prefix_calendar=None, prefix_social=None):
        """
        统一的多步滚动预测：
        - 每次模型仍输出最后一个 token_len
        - 连续滚动 inference_steps 次
        - 最终拼成长度 pred_len 的预测
        """
        inference_steps = pred_len // self.args.token_len
        dis = pred_len - inference_steps * self.args.token_len
        if dis != 0:
            inference_steps += 1

        roll_x = batch_x
        roll_x_mark = batch_x_mark
        pred_y = []

        for j in range(inference_steps):
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(roll_x, roll_x_mark, None, batch_y_mark, prefix_calendar=prefix_calendar, prefix_social=prefix_social)
            else:
                outputs = self.model(roll_x, roll_x_mark, None, batch_y_mark, prefix_calendar=prefix_calendar, prefix_social=prefix_social)

            step_pred = outputs[:, -self.args.token_len:, :]
            pred_y.append(step_pred)

            if j != inference_steps - 1:
                roll_x = torch.cat([roll_x[:, self.args.token_len:, :], step_pred], dim=1)

                # 用未来 mark 的第 j 个 token 继续往前滚
                tmp = batch_y_mark[:, j:j+1, :]
                roll_x_mark = torch.cat([roll_x_mark[:, 1:, :], tmp], dim=1)

        pred_y = torch.cat(pred_y, dim=1)
        
        if dis != 0:
            pred_y = pred_y[:, :pred_len, :]

        return pred_y
    def vali(self, vali_data, vali_loader, criterion, pred_len):
        total_loss = []
        total_count = []
        time_now = time.time()
        test_steps = len(vali_loader)
        iter_count = 0
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, prefix_calendar, prefix_social, meta = self._parse_batch(batch)

                if prefix_calendar is not None:
                    prefix_calendar = prefix_calendar.float().to(self.device)
                if prefix_social is not None:
                    prefix_social = prefix_social.float().to(self.device)
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                pred_y = self._rollout_predict(
                    batch_x=batch_x,
                    batch_x_mark=batch_x_mark,
                    batch_y_mark=batch_y_mark,
                    pred_len=pred_len,
                    prefix_calendar=prefix_calendar,
                    prefix_social=prefix_social
                )

                true_y = batch_y[:, -pred_len:, :]
                loss = criterion(pred_y, true_y)

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])

                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(
                            i + 1, speed, left_time
                        ))
                        iter_count = 0
                        time_now = time.time()

        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)

        self.model.train()
        return total_loss
    def train(self, setting):
        criterion = self._select_criterion()
        train_pred_len = getattr(self.args, "train_pred_len", self.args.token_len)
        def chk(name, t):
            if not torch.isfinite(t).all():
                bad = t[~torch.isfinite(t)]
                print(f"[BAD] {name} shape={tuple(t.shape)} dtype={t.dtype} "
                    f"bad_cnt={bad.numel()} sample={bad.flatten()[:10]}")
                raise RuntimeError(name)

            if t.numel() == 0:
                print(f"[EMPTY] {name} is empty!")
                raise RuntimeError(name)
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        print("len(train_data) =", len(train_data))
        print("len(vali_data)  =", len(vali_data))
        print("len(test_data)  =", len(test_data))
        print("len(train_loader) =", len(train_loader))
        print("len(vali_loader)  =", len(vali_loader))
        print("len(test_loader)  =", len(test_loader))
        path = os.path.join(self.args.checkpoints, setting)
        log_dir = os.path.join(path, "tensorboard")
        writer = SummaryWriter(log_dir)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)
        
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        epoch_logs = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0

            loss_val = torch.tensor(0., device="cuda")
            count = torch.tensor(0., device="cuda")
            
            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, prefix_calendar, prefix_social, meta = self._parse_batch(batch)

                if prefix_calendar is not None:
                    prefix_calendar = prefix_calendar.float().to(self.device)
                if prefix_social is not None:
                    prefix_social = prefix_social.float().to(self.device)
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                chk("batch_x", batch_x)
                chk("batch_y", batch_y)
                chk("batch_x_mark", batch_x_mark)
                chk("batch_y_mark", batch_y_mark)

                # ① mark 基本统计（确认不是全0/常量）
                if i < 3 and ((not self.args.use_multi_gpu) or self.args.local_rank == 0):
                    print("[DEBUG] x_mark mean/std/min/max:",
                        batch_x_mark.mean().item(),
                        batch_x_mark.std().item(),
                        batch_x_mark.min().item(),
                        batch_x_mark.max().item())

                # ② token 差异（确认切片逻辑）
                if i == 0 and ((not self.args.use_multi_gpu) or self.args.local_rank == 0):
                    if batch_x_mark.dim() >= 3 and batch_x_mark.size(1) >= 2:
                        delta = (batch_x_mark[:, 1, :] - batch_x_mark[:, 0, :]).abs().mean().item()
                        print("[DEBUG] mean(|mark[t1]-mark[t0]|) =", delta)

                do_sanity = (i == 0 and epoch == 0 and ((not self.args.use_multi_gpu) or self.args.local_rank == 0))

                pred_y = self._rollout_predict(
                    batch_x=batch_x,
                    batch_x_mark=batch_x_mark,
                    batch_y_mark=batch_y_mark,
                    pred_len=train_pred_len,
                    prefix_calendar=prefix_calendar,
                    prefix_social=prefix_social
                )

                chk("pred_y", pred_y)

                true_y = batch_y[:, -train_pred_len:, :]
                chk("true_y", true_y)

                loss = criterion(pred_y, true_y)
                chk("loss", loss)

                loss_val += loss.detach()
                count += 1

                if do_sanity:
                    D = batch_x_mark.shape[-1]
                    half = D // 2

                    with torch.no_grad():
                        out_full = self._rollout_predict(batch_x, batch_x_mark, batch_y_mark, train_pred_len, prefix_calendar=prefix_calendar, prefix_social=prefix_social)

                        zeros_xm = torch.zeros_like(batch_x_mark)
                        zeros_ym = torch.zeros_like(batch_y_mark)
                        out_zero = self._rollout_predict(batch_x, zeros_xm, zeros_ym, train_pred_len, prefix_calendar=prefix_calendar, prefix_social=prefix_social)

                        xm_time = batch_x_mark.clone()
                        ym_time = batch_y_mark.clone()
                        xm_time[..., half:] = 0
                        ym_time[..., half:] = 0
                        out_time = self._rollout_predict(batch_x, xm_time, ym_time, train_pred_len, prefix_calendar=prefix_calendar, prefix_social=prefix_social)

                        xm_w = batch_x_mark.clone()
                        ym_w = batch_y_mark.clone()
                        xm_w[..., :half] = 0
                        ym_w[..., :half] = 0
                        out_w = self._rollout_predict(batch_x, xm_w, ym_w, train_pred_len, prefix_calendar=prefix_calendar, prefix_social=prefix_social)

                    print("[SANITY] full-vs-zero =", (out_full - out_zero).abs().mean().item())
                    print("[SANITY] full-vs-time =", (out_full - out_time).abs().mean().item(), "  <-- weather contribution")
                    print("[SANITY] full-vs-wthr =", (out_full - out_w).abs().mean().item(), "  <-- time contribution")

                #count += 1

                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    model_optim.step()
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))   
            if self.args.use_multi_gpu:
                dist.barrier()   
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)      
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion, pred_len=train_pred_len)
            test_loss = self.vali(test_data, test_loader, criterion, pred_len=self.args.test_pred_len)
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/vali", vali_loss, epoch)
                writer.add_scalar("loss/test", test_loss, epoch)
                writer.add_scalar("lr", model_optim.param_groups[0]['lr'], epoch)
            log = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "vali_loss": vali_loss,
                "test_loss": test_loss,
                "lr": model_optim.param_groups[0]['lr']
            }
            epoch_logs.append(log)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.use_multi_gpu:
                train_loader.sampler.set_epoch(epoch + 1)
        df = pd.DataFrame(epoch_logs)   
        file_path = "./train_vali_test_loss_per_epoch_2.csv"
        df.to_csv(file_path,mode="a",   header=not os.path.exists(file_path))  # 只有第一次写header index=False)
        writer.close()
        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        else:
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        print("info:", self.args.test_seq_len, self.args.test_label_len, self.args.token_len, self.args.test_pred_len)
        # ===== 记录本次 test 实际使用的 ckpt 路径 =====
        if test:
            print('loading model')
            ckpt_setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            ckpt_path = os.path.join(self.args.checkpoints, ckpt_setting, best_model_path)

            print("loading model from {}".format(ckpt_path))
            load_item = torch.load(ckpt_path)
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)

            # 保持原来的结果文件夹逻辑
            setting = ckpt_setting
        else:
            ckpt_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        print("[TEST META] ckpt_path:", ckpt_path)
        print("[TEST META] time_pt_path:", getattr(self.args, "time_pt_path", ""))
        print("[TEST META] weather_pt_path:", getattr(self.args, "weather_pt_path", ""))
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        self.model.eval()

        did_mark_sanity = False

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, prefix_calendar, prefix_social, meta = self._parse_batch(batch)

                if prefix_calendar is not None:
                    prefix_calendar = prefix_calendar.float().to(self.device)
                if prefix_social is not None:
                    prefix_social = prefix_social.float().to(self.device)

                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # ===== 先打印一下 mark 本身，确认真的读到了 =====
                if i == 0:
                    print("[TEST DEBUG] batch_x_mark shape:", tuple(batch_x_mark.shape))
                    print("[TEST DEBUG] batch_y_mark shape:", tuple(batch_y_mark.shape))
                    print("[TEST DEBUG] x_mark mean/std/min/max:",
                        batch_x_mark.mean().item(),
                        batch_x_mark.std().item(),
                        batch_x_mark.min().item(),
                        batch_x_mark.max().item())
                    print("[TEST DEBUG] y_mark mean/std/min/max:",
                        batch_y_mark.mean().item(),
                        batch_y_mark.std().item(),
                        batch_y_mark.min().item(),
                        batch_y_mark.max().item())

                inference_steps = self.args.test_pred_len // self.args.token_len
                dis = self.args.test_pred_len - inference_steps * self.args.token_len
                if dis != 0:
                    inference_steps += 1

                pred_y = []
                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        batch_x = torch.cat([batch_x[:, self.args.token_len:, :], pred_y[-1]], dim=1)
                        tmp = batch_y_mark[:, j-1:j, :]
                        batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

                    # ===== 只在第一个 batch / 第一步做一次 sanity =====
                    if (not did_mark_sanity) and i == 0 and j == 0:
                        did_mark_sanity = True

                        out_full = self.model(batch_x, batch_x_mark, None, batch_y_mark, prefix_calendar=prefix_calendar, prefix_social=prefix_social)

                        zeros_xm = torch.zeros_like(batch_x_mark)
                        zeros_ym = torch.zeros_like(batch_y_mark)
                        out_zero = self.model(batch_x, zeros_xm, None, zeros_ym, prefix_calendar=prefix_calendar, prefix_social=prefix_social)

                        # 打乱 mark：沿 batch 维打乱，保持 shape 不变
                        perm = torch.randperm(batch_x_mark.size(0), device=batch_x_mark.device)
                        shuf_xm = batch_x_mark[perm]
                        shuf_ym = batch_y_mark[perm]
                        out_shuffle = self.model(batch_x, shuf_xm, None, shuf_ym, prefix_calendar=prefix_calendar, prefix_social=prefix_social)

                        diff_zero = (out_full - out_zero).abs().mean().item()
                        diff_shuffle = (out_full - out_shuffle).abs().mean().item()

                        print("[TEST SANITY] mean(|full - zero_mark|)    =", diff_zero)
                        print("[TEST SANITY] mean(|full - shuffle_mark|) =", diff_shuffle)

                        # 再看看最终真正拿去预测的最后 token 是否变化
                        token_diff_zero = (
                            out_full[:, -self.args.token_len:, :] - out_zero[:, -self.args.token_len:, :]
                        ).abs().mean().item()
                        token_diff_shuffle = (
                            out_full[:, -self.args.token_len:, :] - out_shuffle[:, -self.args.token_len:, :]
                        ).abs().mean().item()

                        print("[TEST SANITY] last-token |full - zero|    =", token_diff_zero)
                        print("[TEST SANITY] last-token |full - shuffle| =", token_diff_shuffle)

                        outputs = out_full
                    else:
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(
                                    batch_x, batch_x_mark, None, batch_y_mark,
                                    prefix_calendar=prefix_calendar, prefix_social=prefix_social
                                )
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, None, batch_y_mark,
                                prefix_calendar=prefix_calendar, prefix_social=prefix_social
                            )

                    pred_y.append(outputs[:, -self.args.token_len:, :])

                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-(self.args.token_len - dis), :]

                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)
                outputs = pred_y.detach().cpu()
                batch_y = batch_y.detach().cpu()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        # ===== 记录本次实验的关键信息到 result txt =====
        time_pt_path = getattr(self.args, "time_pt_path", "")
        weather_pt_path = getattr(self.args, "weather_pt_path", "")
        ms_scale = getattr(self.args, "ms_scale", "")
        mix_embeds = getattr(self.args, "mix_embeds", "")

        with open("result_long_term_forecast.txt", "a", encoding="utf-8") as f:
            f.write("=" * 120 + "\n")
            f.write(f"setting: {setting}\n")
            f.write(f"ckpt_path: {ckpt_path}\n")
            f.write(f"time_pt_path: {time_pt_path}\n")
            f.write(f"weather_pt_path: {weather_pt_path}\n")
            f.write(
                f"seq_len: {self.args.seq_len}, "
                f"label_len: {self.args.label_len}, "
                f"token_len: {self.args.token_len}, "
                f"test_seq_len: {self.args.test_seq_len}, "
                f"test_label_len: {self.args.test_label_len}, "
                f"test_pred_len: {self.args.test_pred_len}\n"
            )
            f.write(
                f"batch_size: {self.args.batch_size}, "
                f"learning_rate: {self.args.learning_rate}, "
                f"weight_decay: {self.args.weight_decay}, "
                f"ms_scale: {ms_scale}, "
                f"mix_embeds: {mix_embeds}\n"
            )
            f.write(f"mse: {mse}, mae: {mae}\n")
            f.write("\n")
    def export_predictions(self, setting, split="test", test=0, save_dir="./xgb_exports", chunk_size=100000):
        """
        导出 AutoTimes 在某个 split(train/val/test) 上的逐样本逐horizon预测明细。
        改为：
        1) parquet 分块写盘
        2) 流式存储，避免 OOM

        输出目录结构：
        save_dir/
            {setting}_{split}_predictions/
                part_00000.parquet
                part_00001.parquet
                ...
        """
        import os
        import gc
        import shutil
        import pandas as pd
        import torch

        assert split in ["train", "val", "test"], f"split must be train/val/test, got {split}"

        # 1) 取数据
        data_set, data_loader = self._get_data(flag=split)

        # 2) 加载 checkpoint
        if test:
            print("loading model for export...")
            ckpt_setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            ckpt_path = os.path.join(self.args.checkpoints, ckpt_setting, best_model_path)

            print("loading model from {}".format(ckpt_path))
            load_item = torch.load(ckpt_path, map_location="cpu")
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)
            setting = ckpt_setting
        else:
            ckpt_path = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
            if os.path.exists(ckpt_path):
                print("loading model from {}".format(ckpt_path))
                load_item = torch.load(ckpt_path, map_location="cpu")
                self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)
            else:
                print("[WARN] checkpoint not found, export will use current in-memory model:", ckpt_path)

        # 3) 确定预测长度
        if split == "test":
            pred_len = self.args.test_pred_len
        else:
            pred_len = getattr(self.args, "train_pred_len", self.args.token_len)

        print(f"[EXPORT] split={split}, pred_len={pred_len}")
        print(f"[EXPORT] ckpt_path={ckpt_path}")

        self.model.eval()
        os.makedirs(save_dir, exist_ok=True)

        # 4) 输出目录：一个 split 一个目录，目录里存多个 parquet part
        out_dir = os.path.join(save_dir, f"{setting}_{split}_predictions")
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        buffer_rows = []
        total_rows = 0
        part_idx = 0
        freq_minutes = getattr(self.args, "freq_minutes", 15)

        def flush_buffer_to_parquet(rows_buffer, current_part_idx):
            if len(rows_buffer) == 0:
                return 0

            chunk_df = pd.DataFrame(rows_buffer)

            # 建议尽量压缩一下，减少磁盘体积
            save_path = os.path.join(out_dir, f"part_{current_part_idx:05d}.parquet")
            chunk_df.to_parquet(save_path, index=False, engine="pyarrow", compression="snappy")

            flushed_rows = len(chunk_df)
            print(f"[EXPORT] flushed {flushed_rows} rows -> {save_path}")

            del chunk_df
            gc.collect()
            return flushed_rows

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, prefix_calendar, prefix_social, meta = self._parse_batch(batch)

                if meta is None:
                    raise ValueError(
                        "meta is None. Please modify Dataset_MultiStation_Custom and data_factory "
                        "so that dataloader returns meta when exporting predictions."
                    )

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if prefix_calendar is not None:
                    prefix_calendar = prefix_calendar.float().to(self.device)
                if prefix_social is not None:
                    prefix_social = prefix_social.float().to(self.device)

                # 用统一 rollout 逻辑得到多步预测
                pred_y = self._rollout_predict(
                    batch_x=batch_x,
                    batch_x_mark=batch_x_mark,
                    batch_y_mark=batch_y_mark,
                    pred_len=pred_len,
                    prefix_calendar=prefix_calendar,
                    prefix_social=prefix_social
                )

                true_y = batch_y[:, -pred_len:, :]

                pred_y = pred_y.detach().cpu().numpy()   # [B, pred_len, 1]
                true_y = true_y.detach().cpu().numpy()   # [B, pred_len, 1]

                # 假设 meta 是 DataLoader collate 后的 dict-of-lists
                sid_list = meta["sid_idx"]
                station_list = meta["station_name"]
                forecast_start_list = meta["forecast_start_time"]

                B = pred_y.shape[0]

                for b in range(B):
                    sid_idx = int(sid_list[b])
                    station_name = str(station_list[b])
                    forecast_start_time = pd.Timestamp(forecast_start_list[b])

                    for h in range(pred_len):
                        target_time = forecast_start_time + pd.Timedelta(minutes=freq_minutes * h)

                        yhat = float(pred_y[b, h, 0])
                        ytrue = float(true_y[b, h, 0])

                        # 尽量只保留训练 XGBoost 真正需要的列
                        buffer_rows.append({
                            "split": split,
                            "sid_idx": sid_idx,
                            "station_name": station_name,
                            "forecast_start_time": forecast_start_time,
                            "target_time": target_time,
                            "horizon": h + 1,
                            "y_true": ytrue,
                            "y_hat_T": yhat,
                            "residual": ytrue - yhat,
                        })

                # 到达 chunk_size 就立刻写一个 parquet part
                if len(buffer_rows) >= chunk_size:
                    flushed = flush_buffer_to_parquet(buffer_rows, part_idx)
                    total_rows += flushed
                    part_idx += 1
                    buffer_rows.clear()

                if (i + 1) % 100 == 0:
                    print(f"[EXPORT] processed {i + 1}/{len(data_loader)} batches")

                # 显式释放一部分中间变量
                del pred_y, true_y
                gc.collect()

        # 最后剩余的再写一次
        if len(buffer_rows) > 0:
            flushed = flush_buffer_to_parquet(buffer_rows, part_idx)
            total_rows += flushed
            part_idx += 1
            buffer_rows.clear()

        # 额外保存一个简单的 meta 文件，便于后续检查
        meta_info = pd.DataFrame([{
            "setting": setting,
            "split": split,
            "ckpt_path": ckpt_path,
            "pred_len": pred_len,
            "freq_minutes": freq_minutes,
            "num_parts": part_idx,
            "total_rows": total_rows,
        }])
        meta_info.to_parquet(
            os.path.join(out_dir, "_meta.parquet"),
            index=False,
            engine="pyarrow",
            compression="snappy"
        )

        print(f"[EXPORT DONE] total_rows={total_rows}")
        print(f"[EXPORT DONE] num_parts={part_idx}")
        print(f"[EXPORT DONE] saved dir: {out_dir}")

        return out_dir
    def export_xgb_ready_dataset(
        self,
        setting,
        split="train",
        test=0,
        save_dir="./xgb_ready_exports",
        chunk_size=100000
    ):
        """
        直接导出适合 XGBoost 训练的 parquet part 文件。
        每行就是一个 horizon 样本，包含：
        - AutoTimes 输出: y_hat_T
        - 标签: label = y_true - y_hat_T
        - lag / rolling / calendar 特征
        - 基础索引: sid_idx, horizon

        重要：
        - 所有 lag / rolling 只使用 forecast origin (= s_end) 之前的历史
        - 不允许使用 target_idx 对应时刻之前的真实值，否则会泄露未来信息
        """
        import os
        import gc
        import shutil
        import numpy as np
        import pandas as pd
        import torch

        assert split in ["train", "val", "test"], f"split must be train/val/test, got {split}"

        data_set, data_loader = self._get_data(flag=split)

        # 加载 checkpoint
        if test:
            print("loading model for xgb-ready export...")
            ckpt_setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            ckpt_path = os.path.join(self.args.checkpoints, ckpt_setting, best_model_path)
            print("loading model from {}".format(ckpt_path))
            load_item = torch.load(ckpt_path, map_location="cpu")
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)
            setting = ckpt_setting
        else:
            ckpt_path = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
            if os.path.exists(ckpt_path):
                print("loading model from {}".format(ckpt_path))
                load_item = torch.load(ckpt_path, map_location="cpu")
                self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)
            else:
                print("[WARN] checkpoint not found, export will use current in-memory model:", ckpt_path)

        if split == "test":
            pred_len = self.args.test_pred_len
        else:
            pred_len = getattr(self.args, "train_pred_len", self.args.token_len)

        print(f"[XGB-READY EXPORT] split={split}, pred_len={pred_len}")
        print(f"[XGB-READY EXPORT] ckpt_path={ckpt_path}")

        self.model.eval()
        os.makedirs(save_dir, exist_ok=True)

        out_dir = os.path.join(save_dir, f"{setting}_{split}_xgb_ready")
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        buffer_rows = []
        total_rows = 0
        part_idx = 0

        def _calendar_feats(ts: pd.Timestamp):
            return {
                "hour": int(ts.hour),
                "minute": int(ts.minute),
                "day_of_week": int(ts.dayofweek),
                "month": int(ts.month),
                "day": int(ts.day),
                "is_weekend": int(ts.dayofweek >= 5),
            }

        def _safe_lag_and_roll(y_hist: np.ndarray, origin_idx: int):
            """
            y_hist: 某个站点全量历史序列, shape [T]
            origin_idx: forecast origin 在全局 dt 中的位置
                        所有 lag / rolling 都只能使用 [:origin_idx] 的历史
            """
            def lag(k):
                idx = origin_idx - k
                if idx < 0:
                    return np.nan
                return float(y_hist[idx])

            def roll_mean(win):
                l = origin_idx - win
                r = origin_idx
                if l < 0:
                    return np.nan
                arr = y_hist[l:r]
                if arr.size == 0:
                    return np.nan
                return float(np.mean(arr))

            def roll_std(win):
                l = origin_idx - win
                r = origin_idx
                if l < 0:
                    return np.nan
                arr = y_hist[l:r]
                if arr.size == 0:
                    return np.nan
                return float(np.std(arr))

            return {
                "lag_1": lag(1),
                "lag_4": lag(4),
                "lag_96": lag(96),
                "lag_672": lag(672),
                "rolling_mean_4": roll_mean(4),
                "rolling_std_4": roll_std(4),
                "rolling_mean_96": roll_mean(96),
                "rolling_std_96": roll_std(96),
            }

        def flush_buffer(rows_buffer, current_part_idx):
            if len(rows_buffer) == 0:
                return 0
            chunk_df = pd.DataFrame(rows_buffer)
            save_path = os.path.join(out_dir, f"part_{current_part_idx:05d}.parquet")
            chunk_df.to_parquet(save_path, index=False, engine="pyarrow", compression="snappy")
            flushed = len(chunk_df)
            print(f"[XGB-READY EXPORT] flushed {flushed} rows -> {save_path}")
            del chunk_df
            gc.collect()
            return flushed

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, prefix_calendar, prefix_social, meta = self._parse_batch(batch)

                if meta is None:
                    raise ValueError("meta is None. export_xgb_ready_dataset requires return_meta=True")

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if prefix_calendar is not None:
                    prefix_calendar = prefix_calendar.float().to(self.device)
                if prefix_social is not None:
                    prefix_social = prefix_social.float().to(self.device)

                pred_y = self._rollout_predict(
                    batch_x=batch_x,
                    batch_x_mark=batch_x_mark,
                    batch_y_mark=batch_y_mark,
                    pred_len=pred_len,
                    prefix_calendar=prefix_calendar,
                    prefix_social=prefix_social
                )

                true_y = batch_y[:, -pred_len:, :]

                pred_y = pred_y.detach().cpu().numpy()   # [B, pred_len, 1]
                true_y = true_y.detach().cpu().numpy()   # [B, pred_len, 1]

                sid_list = meta["sid_idx"]
                s_end_list = meta["s_end"]

                B = pred_y.shape[0]

                for b in range(B):
                    sid_idx = int(sid_list[b])
                    s_end = int(s_end_list[b])

                    # forecast origin：所有特征只能基于这个时间点之前的历史
                    origin_idx = s_end

                    # 全量历史序列（已经按全局 dt 对齐）
                    y_hist = data_set.Y[sid_idx, :, 0]

                    # 同一个 forecast origin 下，lag / rolling 固定不变
                    hist_feats = _safe_lag_and_roll(y_hist, origin_idx)

                    for h in range(pred_len):
                        target_idx = s_end + h
                        if target_idx >= len(data_set.dt):
                            continue

                        target_time = pd.Timestamp(data_set.dt[target_idx])

                        row = {
                            "sid_idx": sid_idx,
                            "horizon": h + 1,
                            "y_true": float(true_y[b, h, 0]),
                            "y_hat_T": float(pred_y[b, h, 0]),
                            "label": float(true_y[b, h, 0] - pred_y[b, h, 0]),
                        }

                        row.update(_calendar_feats(target_time))
                        row.update(hist_feats)

                        buffer_rows.append(row)

                if len(buffer_rows) >= chunk_size:
                    flushed = flush_buffer(buffer_rows, part_idx)
                    total_rows += flushed
                    part_idx += 1
                    buffer_rows.clear()

                if (i + 1) % 100 == 0:
                    print(f"[XGB-READY EXPORT] processed {i + 1}/{len(data_loader)} batches")

                del pred_y, true_y
                gc.collect()

        if len(buffer_rows) > 0:
            flushed = flush_buffer(buffer_rows, part_idx)
            total_rows += flushed
            part_idx += 1
            buffer_rows.clear()

        meta_info = pd.DataFrame([{
            "setting": setting,
            "split": split,
            "ckpt_path": ckpt_path,
            "pred_len": pred_len,
            "num_parts": part_idx,
            "total_rows": total_rows,
        }])
        meta_info.to_parquet(
            os.path.join(out_dir, "_meta.parquet"),
            index=False,
            engine="pyarrow",
            compression="snappy"
        )

        print(f"[XGB-READY EXPORT DONE] total_rows={total_rows}")
        print(f"[XGB-READY EXPORT DONE] num_parts={part_idx}")
        print(f"[XGB-READY EXPORT DONE] saved dir: {out_dir}")

        return out_dir
    def _parse_batch(self, batch):
        """
        统一解析 dataloader 返回的 batch
        支持：
        - 4项: x, y, x_mark, y_mark
        - 5项: x, y, x_mark, y_mark, prefix_calendar
        - 6项: x, y, x_mark, y_mark, prefix_calendar, prefix_social
        - 7项: x, y, x_mark, y_mark, prefix_calendar, prefix_social, meta
        """
        meta = None

        if len(batch) == 7:
            batch_x, batch_y, batch_x_mark, batch_y_mark, prefix_calendar, prefix_social, meta = batch
        elif len(batch) == 6:
            batch_x, batch_y, batch_x_mark, batch_y_mark, prefix_calendar, prefix_social = batch
        elif len(batch) == 5:
            batch_x, batch_y, batch_x_mark, batch_y_mark, prefix_calendar = batch
            prefix_social = None
        elif len(batch) == 4:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            prefix_calendar = None
            prefix_social = None
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")

        return batch_x, batch_y, batch_x_mark, batch_y_mark, prefix_calendar, prefix_social, meta
    def export_all_predictions(self, setting, test=0, save_dir="./xgb_exports", chunk_size=100000):
        """
        用同一个模型/ckpt，连续导出 train / val / test 三个 split。
        """
        results = {}
        for split in ["train", "val", "test"]:
            print(f"\n[EXPORT ALL] start split = {split}")
            out_dir = self.export_predictions(
                setting=setting,
                split=split,
                test=test,
                save_dir=save_dir,
                chunk_size=chunk_size
            )
            results[split] = out_dir
            print(f"[EXPORT ALL] done split = {split}, saved to {out_dir}")
        return results
    def export_all_xgb_ready_datasets(self, setting, test=0, save_dir="./xgb_ready_exports", chunk_size=100000):
        results = {}
        for split in ["train", "val", "test"]:
            print(f"\n[XGB-READY EXPORT ALL] start split = {split}")
            out_dir = self.export_xgb_ready_dataset(
                setting=setting,
                split=split,
                test=test,
                save_dir=save_dir,
                chunk_size=chunk_size
            )
            results[split] = out_dir
            print(f"[XGB-READY EXPORT ALL] done split = {split}, saved to {out_dir}")
        return results
    def test_with_xgb(self, setting, xgb_model_path, test=0):
        """
        用 AutoTimes + XGBoost residual correction 按原始 dataloader/test 口径评估。
        口径与原始 test() 一致：
        - 直接在 dataloader 的 batch 上 rollout
        - 不导出 parquet
        - 不 merge raw csv
        - 不 dedup
        - 最后统一 metric(preds, trues)

        XGBoost 预测的是 residual:
            residual_hat = f(features)
        最终预测:
            pred_final = y_hat_T + residual_hat
        """
        import xgboost as xgb
        import pandas as pd
        import numpy as np
        import torch
        from utils.metrics import metric

        test_data, test_loader = self._get_data(flag='test')

        print("info:", self.args.test_seq_len, self.args.test_label_len, self.args.token_len, self.args.test_pred_len)

        # ========= 1) 加载 AutoTimes checkpoint =========
        if test:
            print('loading model')
            ckpt_setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            ckpt_path = os.path.join(self.args.checkpoints, ckpt_setting, best_model_path)

            print("loading model from {}".format(ckpt_path))
            load_item = torch.load(ckpt_path, map_location="cpu")
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)

            setting = ckpt_setting
        else:
            ckpt_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            if os.path.exists(ckpt_path):
                print("loading model from {}".format(ckpt_path))
                load_item = torch.load(ckpt_path, map_location="cpu")
                self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)
            else:
                print("[WARN] checkpoint not found, will use current in-memory model:", ckpt_path)

        # ========= 2) 加载 XGBoost =========
        booster = xgb.Booster()
        booster.load_model(xgb_model_path)
        print("[XGB] loaded model from", xgb_model_path)

        preds_base = []
        preds_final = []
        trues = []

        folder_path = './test_results_xgb/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0

        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, prefix_calendar, prefix_social, meta = self._parse_batch(batch)

                if prefix_calendar is not None:
                    prefix_calendar = prefix_calendar.float().to(self.device)
                if prefix_social is not None:
                    prefix_social = prefix_social.float().to(self.device)

                iter_count += 1

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # ========= 3) AutoTimes rollout =========
                pred_y = self._rollout_predict(
                    batch_x=batch_x,
                    batch_x_mark=batch_x_mark,
                    batch_y_mark=batch_y_mark,
                    pred_len=self.args.test_pred_len,
                    prefix_calendar=prefix_calendar,
                    prefix_social=prefix_social
                )  # [B, H, 1]

                true_y = batch_y[:, -self.args.test_pred_len:, :]  # [B, H, 1]

                pred_y_np = pred_y.detach().cpu().numpy()
                true_y_np = true_y.detach().cpu().numpy()

                # ========= 4) 构造 XGBoost 输入 =========
                # 当前最基础版本只用:
                # sid_idx, horizon, y_hat_T
                if meta is None:
                    raise ValueError("meta is None. test_with_xgb requires meta with sid_idx.")

                sid_list = meta["sid_idx"]

                B, H, C = pred_y_np.shape
                assert C == 1, f"Expected last dim = 1, got {C}"

                rows = []
                for b in range(B):
                    sid_idx = int(sid_list[b])
                    for h in range(H):
                        rows.append({
                            "sid_idx": sid_idx,
                            "horizon": h + 1,
                            "y_hat_T": float(pred_y_np[b, h, 0]),
                        })

                X_df = pd.DataFrame(rows)

                # ========= 5) XGBoost 预测 residual，并做修正 =========
                pred_residual = booster.predict(xgb.DMatrix(X_df))   # [B*H]
                pred_residual = pred_residual.reshape(B, H, 1)

                pred_final_np = pred_y_np + pred_residual

                preds_base.append(pred_y_np)
                preds_final.append(pred_final_np)
                trues.append(true_y_np)

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (test_steps - i)
                    print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                    iter_count = 0
                    time_now = time.time()

        # ========= 6) 仿照原始 test()：统一拼接，统一计算 metric =========
        preds_base = np.concatenate(preds_base, axis=0)
        preds_final = np.concatenate(preds_final, axis=0)
        trues = np.concatenate(trues, axis=0)

        mae_base, mse_base, rmse_base, mape_base, mspe_base = metric(preds_base, trues)
        mae_final, mse_final, rmse_final, mape_final, mspe_final = metric(preds_final, trues)

        print('[BASE] mse:{}, mae:{}'.format(mse_base, mae_base))
        print('[XGB ] mse:{}, mae:{}'.format(mse_final, mae_final))

        # ========= 7) 写结果 =========
        with open("result_long_term_forecast_xgb.txt", "a", encoding="utf-8") as f:
            f.write("=" * 120 + "\n")
            f.write(f"setting: {setting}\n")
            f.write(f"ckpt_path: {ckpt_path}\n")
            f.write(f"xgb_model_path: {xgb_model_path}\n")
            f.write(f"time_pt_path: {getattr(self.args, 'time_pt_path', '')}\n")
            f.write(f"weather_pt_path: {getattr(self.args, 'weather_pt_path', '')}\n")
            f.write(
                f"seq_len: {self.args.seq_len}, "
                f"label_len: {self.args.label_len}, "
                f"token_len: {self.args.token_len}, "
                f"test_pred_len: {self.args.test_pred_len}\n"
            )
            f.write(
                f"[BASE] mse: {mse_base}, mae: {mae_base}, rmse: {rmse_base}, mape: {mape_base}, mspe: {mspe_base}\n"
            )
            f.write(
                f"[XGB ] mse: {mse_final}, mae: {mae_final}, rmse: {rmse_final}, mape: {mape_final}, mspe: {mspe_final}\n"
            )
            f.write("\n")

        return {
            "base": {
                "mse": float(mse_base),
                "mae": float(mae_base),
                "rmse": float(rmse_base),
                "mape": float(mape_base),
                "mspe": float(mspe_base),
            },
            "xgb": {
                "mse": float(mse_final),
                "mae": float(mae_final),
                "rmse": float(rmse_final),
                "mape": float(mape_final),
                "mspe": float(mspe_final),
            }
        }