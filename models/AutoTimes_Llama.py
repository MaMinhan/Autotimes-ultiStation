import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from layers.mlp import MLP


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len

        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)

        self.llama = LlamaForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32,
        )

        # 原代码写死 4096，这里也保留你的设定
        # 若你后面想更稳，可改成 self.llama.config.hidden_size
        self.hidden_dim_of_llama = 4096

        self.mix = configs.mix_embeds
        print("[MODEL INIT] mix_embeds =", self.mix)

        # ===== 按 GPT2 版新增 =====
        self.mark_input_dim = getattr(configs, "mark_input_dim", self.hidden_dim_of_llama)

        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))

        if self.mark_input_dim == self.hidden_dim_of_llama:
            # only time.pt
            self.use_weather = False
            self.time_proj = nn.Linear(self.hidden_dim_of_llama, self.hidden_dim_of_llama)
            self.mark_fuse = nn.Identity()

        elif self.mark_input_dim == self.hidden_dim_of_llama * 2:
            # time.pt + weather.pt
            self.use_weather = True

            # 各自独立保留完整语义空间
            self.time_proj = nn.Linear(self.hidden_dim_of_llama, self.hidden_dim_of_llama)
            self.weather_proj = nn.Linear(self.hidden_dim_of_llama, self.hidden_dim_of_llama)

            # 先做各自归一化，减少数值尺度偏移
            self.time_norm = nn.LayerNorm(self.hidden_dim_of_llama)
            self.weather_norm = nn.LayerNorm(self.hidden_dim_of_llama)
            self.fused_norm = nn.LayerNorm(self.hidden_dim_of_llama)

            # 门控：根据 times_embeds + time + weather 决定每个维度保留多少
            self.gate_net = nn.Sequential(
                nn.Linear(self.hidden_dim_of_llama * 3, self.hidden_dim_of_llama * 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim_of_llama * 2, self.hidden_dim_of_llama * 3),
                nn.Sigmoid()
            )

            # 最终融合时保留：
            # time_feat / weather_feat / time*weather / gated_sum
            # 总共 4H -> H
            self.mark_fuse = nn.Sequential(
                nn.Linear(self.hidden_dim_of_llama * 4, self.hidden_dim_of_llama * 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim_of_llama * 2, self.hidden_dim_of_llama),
            )
        else:
            raise ValueError(
                f"Unsupported mark_input_dim={self.mark_input_dim}. "
                f"Expected {self.hidden_dim_of_llama} or {self.hidden_dim_of_llama * 2}."
            )
        print("mark_input_dim =", self.mark_input_dim)
        print("use_weather =", getattr(self, "use_weather", None))
        for name, param in self.llama.named_parameters():
            param.requires_grad = False

        if configs.mlp_hidden_layers == 0:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_llama)
            self.decoder = nn.Linear(self.hidden_dim_of_llama, self.token_len)
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(
                self.token_len,
                self.hidden_dim_of_llama,
                configs.mlp_hidden_dim,
                configs.mlp_hidden_layers,
                configs.dropout,
                configs.mlp_activation
            )
            self.decoder = MLP(
                self.hidden_dim_of_llama,
                self.token_len,
                configs.mlp_hidden_dim,
                configs.mlp_hidden_layers,
                configs.dropout,
                configs.mlp_activation
            )
                    # ===== Multi-scale token =====
            self.use_multiscale = getattr(configs, "use_multiscale", False)
            self.ms_fusion = getattr(configs, "ms_fusion", "sum")   # sum / weighted
            self.ms_pattern_pool = getattr(configs, "ms_pattern_pool", 4)  # 4个15min=1h

            if self.use_multiscale:
                if self.token_len % self.ms_pattern_pool != 0:
                    raise ValueError(
                        f"token_len={self.token_len} must be divisible by "
                        f"ms_pattern_pool={self.ms_pattern_pool}"
                    )

                self.pattern_len = self.token_len // self.ms_pattern_pool
                self.rr_len = (self.token_len - 1) + self.token_len   # ramp(95) + residual(96) = 191 when token_len=96

                # fine: 原始 96 点 -> hidden
                self.fine_encoder = nn.Linear(self.token_len, self.hidden_dim_of_llama)

                # pattern: 24 点 -> hidden
                self.pattern_encoder = nn.Linear(self.pattern_len, self.hidden_dim_of_llama)

                # rr: 191 点 -> hidden
                self.rr_encoder = nn.Linear(self.rr_len, self.hidden_dim_of_llama)

                if self.ms_fusion == "weighted":
                    self.alpha_fine = nn.Parameter(torch.tensor(0.0))
                    self.alpha_pattern = nn.Parameter(torch.tensor(0.0))
                    self.alpha_rr = nn.Parameter(torch.tensor(0.0))

            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("[MODEL INIT] use_multiscale =", self.use_multiscale)
                if self.use_multiscale:
                    print("[MODEL INIT] ms_fusion =", self.ms_fusion)
                    print("[MODEL INIT] pattern_len =", self.pattern_len)
                    #print("[MODEL INIT] rr_len =", self.rr_len)
    def build_multiscale_embeds(self, fold_out):
        """
        fold_out: [B*C, token_num, token_len]
        return:   [B*C, token_num, hidden_dim]
        """
        # fine token: 原始 96 点
        fine = fold_out  # [BC, T, 96]

        # pattern token: 每 4 点聚合 -> 24 点
        p = self.ms_pattern_pool
        pattern = fold_out.reshape(
            fold_out.shape[0], fold_out.shape[1], self.pattern_len, p
        ).mean(dim=-1)  # [BC, T, 24]

        # ramp token: 平滑后的一阶差分
        # replicate pad on last dimension
        x_pad = torch.nn.functional.pad(fold_out, (1, 1), mode='replicate')
        x_smooth = (x_pad[..., :-2] + x_pad[..., 1:-1] + x_pad[..., 2:]) / 3.0
        ramp = x_smooth[..., 1:] - x_smooth[..., :-1]  # [BC, T, 95]

        # residual token: 原始 - pattern上采样
        pattern_up = pattern.repeat_interleave(p, dim=-1)  # [BC, T, 96]
        residual = fold_out - pattern_up                   # [BC, T, 96]

        rr = torch.cat([ramp, residual], dim=-1)          # [BC, T, 191]

        z_fine = self.fine_encoder(fine)
        z_pattern = self.pattern_encoder(pattern)
        z_rr = self.rr_encoder(rr)

        if self.ms_fusion == "sum":
            z = z_fine + z_pattern + z_rr
        elif self.ms_fusion == "weighted":
            z = (
                self.alpha_fine * z_fine
                + self.alpha_pattern * z_pattern
                + self.alpha_rr * z_rr
            )
        else:
            raise ValueError(f"Unsupported ms_fusion={self.ms_fusion}")

        return z
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        bs, _, n_vars = x_enc.shape

        # [B, L, C] -> [B, C, L]
        x_enc = x_enc.permute(0, 2, 1)

        # [B*C, L]
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)

        # [B*C, token_num, token_len]
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = fold_out.shape[1]

        if self.use_multiscale:
            times_embeds = self.build_multiscale_embeds(fold_out)
        else:
            times_embeds = self.encoder(fold_out)

        if self.mix:
            if x_mark_enc is None:
                raise ValueError("mix_embeds=True but x_mark_enc is None")

            # 和 GPT2 版一致：先对 times_embeds 做归一化
            times_embeds = times_embeds / (times_embeds.norm(dim=2, keepdim=True) + 1e-8)

            if not getattr(self, "use_weather", False):
                # only time.pt
                time_mark = self.time_proj(x_mark_enc)
                time_mark = time_mark / (time_mark.norm(dim=2, keepdim=True) + 1e-8)
                fused_mark = self.mark_fuse(time_mark)
 
            else:
                # time.pt + weather.pt
                time_mark = x_mark_enc[..., :self.hidden_dim_of_llama]
                weather_mark = x_mark_enc[..., self.hidden_dim_of_llama:]

                # 1) 各自独立投影，先不压缩
                time_feat = self.time_proj(time_mark)
                weather_feat = self.weather_proj(weather_mark)

                # 2) norm + unit normalization，尽量稳定
                time_feat = self.time_norm(time_feat)
                weather_feat = self.weather_norm(weather_feat)

                time_feat = time_feat / (time_feat.norm(dim=2, keepdim=True) + 1e-8)
                weather_feat = weather_feat / (weather_feat.norm(dim=2, keepdim=True) + 1e-8)

                # 3) 用 times_embeds + time_feat + weather_feat 共同决定门控
                gate_in = torch.cat([times_embeds, time_feat, weather_feat], dim=-1)
                gates = self.gate_net(gate_in)

                g_base, g_time, g_weather = torch.chunk(gates, 3, dim=-1)

                # 4) gated sum：不是粗暴平均，而是逐维决定保留程度
                gated_sum = (
                    g_base * times_embeds +
                    g_time * time_feat +
                    g_weather * weather_feat
                )

                # 5) 显式加入交互项，保留 time-weather 耦合信息
                interaction = time_feat * weather_feat

                # 6) 最终融合：4H -> H
                fuse_in = torch.cat(
                    [time_feat, weather_feat, interaction, gated_sum],
                    dim=-1
                )
                fused_mark = self.mark_fuse(fuse_in)
                fused_mark = self.fused_norm(fused_mark)
                fused_mark = fused_mark / (fused_mark.norm(dim=2, keepdim=True) + 1e-8)

            if times_embeds.shape != fused_mark.shape:
                raise RuntimeError(
                    f"shape mismatch after fusion: times_embeds={times_embeds.shape}, "
                    f"fused_mark={fused_mark.shape}"
                )

            times_embeds = times_embeds + self.add_scale * fused_mark

        # 和你原 Llama 版一致
        # 进入 Llama 前：转成 llama 的 dtype/device
        llama_param = next(self.llama.parameters())
        llama_device = llama_param.device
        llama_dtype = llama_param.dtype
        times_embeds = times_embeds.to(device=llama_device, dtype=llama_dtype)

        # Llama forward
        outputs = self.llama.model(inputs_embeds=times_embeds)[0]

        # 出来后：转成 decoder 的 dtype/device
        decoder_param = next(self.decoder.parameters())
        outputs = outputs.to(device=decoder_param.device, dtype=decoder_param.dtype)

        dec_out = self.decoder(outputs)

        dec_out = dec_out.reshape(bs, n_vars, -1)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1)
        )

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)