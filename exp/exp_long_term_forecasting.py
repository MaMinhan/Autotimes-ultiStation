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
    def _rollout_predict(self, batch_x, batch_x_mark, batch_y_mark, pred_len):
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
                    outputs = self.model(roll_x, roll_x_mark, None, batch_y_mark)
            else:
                outputs = self.model(roll_x, roll_x_mark, None, batch_y_mark)

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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                pred_y = self._rollout_predict(
                    batch_x=batch_x,
                    batch_x_mark=batch_x_mark,
                    batch_y_mark=batch_y_mark,
                    pred_len=pred_len
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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
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
                    pred_len=train_pred_len
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
                        out_full = self._rollout_predict(batch_x, batch_x_mark, batch_y_mark, train_pred_len)

                        zeros_xm = torch.zeros_like(batch_x_mark)
                        zeros_ym = torch.zeros_like(batch_y_mark)
                        out_zero = self._rollout_predict(batch_x, zeros_xm, zeros_ym, train_pred_len)

                        xm_time = batch_x_mark.clone()
                        ym_time = batch_y_mark.clone()
                        xm_time[..., half:] = 0
                        ym_time[..., half:] = 0
                        out_time = self._rollout_predict(batch_x, xm_time, ym_time, train_pred_len)

                        xm_w = batch_x_mark.clone()
                        ym_w = batch_y_mark.clone()
                        xm_w[..., :half] = 0
                        ym_w[..., :half] = 0
                        out_w = self._rollout_predict(batch_x, xm_w, ym_w, train_pred_len)

                    print("[SANITY] full-vs-zero =", (out_full - out_zero).abs().mean().item())
                    print("[SANITY] full-vs-time =", (out_full - out_time).abs().mean().item(), "  <-- weather contribution")
                    print("[SANITY] full-vs-wthr =", (out_full - out_w).abs().mean().item(), "  <-- time contribution")

                count += 1

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
        df.to_csv(file_path,mode="a",   header=not os.path.exists(file_path)),  # 只有第一次写header index=False)
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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
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

                        out_full = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                        zeros_xm = torch.zeros_like(batch_x_mark)
                        zeros_ym = torch.zeros_like(batch_y_mark)
                        out_zero = self.model(batch_x, zeros_xm, None, zeros_ym)

                        # 打乱 mark：沿 batch 维打乱，保持 shape 不变
                        perm = torch.randperm(batch_x_mark.size(0), device=batch_x_mark.device)
                        shuf_xm = batch_x_mark[perm]
                        shuf_ym = batch_y_mark[perm]
                        out_shuffle = self.model(batch_x, shuf_xm, None, shuf_ym)

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
                                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

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