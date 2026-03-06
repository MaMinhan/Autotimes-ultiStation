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

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
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
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                else:
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                if is_test:
                    outputs = outputs[:, -self.args.token_len:, :]
                    batch_y = batch_y[:, -self.args.token_len:, :].to(self.device)
                else:
                    outputs = outputs[:, :, :]
                    batch_y = batch_y[:, :, :].to(self.device)

                loss = criterion(outputs, batch_y)

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
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
        if getattr(self.args, "resume", False):
            if not self.args.resume_ckpt:
                raise ValueError("resume=True 但没有提供 --resume_ckpt")
            print(f"[RESUME] loading checkpoint from {self.args.resume_ckpt}")
            load_item = torch.load(self.args.resume_ckpt, map_location=self.device)

            # 兼容 DDP / 非DDP
            if isinstance(load_item, dict):
                try:
                    self.model.load_state_dict(load_item, strict=False)
                except RuntimeError:
                    self.model.load_state_dict(
                        {k.replace('module.', ''): v for k, v in load_item.items()},
                        strict=False
                    )
            else:
                raise ValueError("checkpoint 格式异常")
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

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                        chk("outputs", outputs)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    chk("outputs", outputs)
                    loss = criterion(outputs, batch_y)

                chk("loss", loss)

                if do_sanity:
                    D = batch_x_mark.shape[-1]
                    half = D // 2  # 1536->768
                    with torch.no_grad():
                        # 0) full
                        out_full = outputs

                        # 1) zero all mark
                        zeros_xm = torch.zeros_like(batch_x_mark)
                        zeros_ym = torch.zeros_like(batch_y_mark)
                        out_zero = self.model(batch_x, zeros_xm, None, zeros_ym)

                        # 2) time-only: keep first half, zero second half (weather)
                        xm_time = batch_x_mark.clone()
                        ym_time = batch_y_mark.clone()
                        xm_time[..., half:] = 0
                        ym_time[..., half:] = 0
                        out_time = self.model(batch_x, xm_time, None, ym_time)

                        # 3) weather-only: keep second half, zero first half (time)
                        xm_w = batch_x_mark.clone()
                        ym_w = batch_y_mark.clone()
                        xm_w[..., :half] = 0
                        ym_w[..., :half] = 0
                        out_w = self.model(batch_x, xm_w, None, ym_w)

                    print("[SANITY] full-vs-zero =", (out_full - out_zero).abs().mean().item())
                    print("[SANITY] full-vs-time =", (out_full - out_time).abs().mean().item(), "  <-- weather contribution")
                    print("[SANITY] full-vs-wthr =", (out_full - out_w).abs().mean().item(), "  <-- time contribution")
                
                loss_val += loss.item()
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

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion, is_test=True)
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/vali", vali_loss, epoch)
                writer.add_scalar("loss/test", test_loss, epoch)
                writer.add_scalar("lr", model_optim.param_groups[0]['lr'], epoch)
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
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name

            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            load_item = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if i == 0:
                    print(f"[TEST] batch_x.shape={batch_x.shape}")
                    print(f"[TEST] batch_y.shape={batch_y.shape}")
                    print(f"[TEST] batch_x_mark.shape={batch_x_mark.shape}")
                    print(f"[TEST] batch_y_mark.shape={batch_y_mark.shape}")
                    print(f"[TEST] token_len={self.args.token_len}, test_pred_len={self.args.test_pred_len}")
                inference_steps = self.args.test_pred_len // self.args.token_len
                dis = self.args.test_pred_len - inference_steps * self.args.token_len
                if dis != 0:
                    inference_steps += 1
                if i == 0:
                    print(f"[TEST] inference_steps={inference_steps}, dis={dis}")
                pred_y = []
                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        batch_x = torch.cat([batch_x[:, self.args.token_len:, :], pred_y[-1]], dim=1)
                        tmp = batch_y_mark[:, j-1:j, :]
                        batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    step_pred = outputs[:, -self.args.token_len:, :]
                    pred_y.append(step_pred)
                    if i == 0:
                        print(f"[TEST] step={j}, step_pred.shape={step_pred.shape}")
                pred_y = torch.cat(pred_y, dim=1)
                if i == 0:
                    print(f"[TEST] pred_y before trim={pred_y.shape}")
                if dis != 0:
                    pred_y = pred_y[:, :-(self.args.token_len - dis), :]
                if i == 0:
                    print(f"[TEST] pred_y after trim={pred_y.shape}")
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)
                if i == 0:
                    print(f"[TEST] aligned batch_y={batch_y.shape}")
                outputs = pred_y.detach().cpu()
                batch_y = batch_y.detach().cpu()
                pred = outputs
                true = batch_y
                if i == 0:
                    print(f"[TEST] final pred.shape={pred.shape}, true.shape={true.shape}")
        
        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        return
