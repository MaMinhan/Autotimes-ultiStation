import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
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

        self.gpt2 = GPT2Model.from_pretrained(configs.llm_ckp_dir)
        self.hidden_dim_of_gpt2 = 768

        self.mix = configs.mix_embeds
        print("[MODEL INIT] mix_embeds =", self.mix)

        self.mark_input_dim = getattr(configs, "mark_input_dim", self.hidden_dim_of_gpt2)
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))

            if self.mark_input_dim == self.hidden_dim_of_gpt2:
                # only time.pt OR single fused mark
                self.use_weather = False
                self.time_proj = nn.Linear(self.hidden_dim_of_gpt2, self.hidden_dim_of_gpt2)
                self.mark_fuse = nn.Identity()

            elif self.mark_input_dim == self.hidden_dim_of_gpt2 * 2:
                # time.pt + weather.pt
                self.use_weather = True

                self.time_proj = nn.Linear(self.hidden_dim_of_gpt2, self.hidden_dim_of_gpt2)
                self.weather_proj = nn.Linear(self.hidden_dim_of_gpt2, self.hidden_dim_of_gpt2)

                self.time_norm = nn.LayerNorm(self.hidden_dim_of_gpt2)
                self.weather_norm = nn.LayerNorm(self.hidden_dim_of_gpt2)
                self.fused_norm = nn.LayerNorm(self.hidden_dim_of_gpt2)

                self.gate_net = nn.Sequential(
                    nn.Linear(self.hidden_dim_of_gpt2 * 3, self.hidden_dim_of_gpt2 * 2),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim_of_gpt2 * 2, self.hidden_dim_of_gpt2 * 3),
                    nn.Sigmoid()
                )

                self.mark_fuse = nn.Sequential(
                    nn.Linear(self.hidden_dim_of_gpt2 * 4, self.hidden_dim_of_gpt2 * 2),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim_of_gpt2 * 2, self.hidden_dim_of_gpt2),
                )
            else:
                raise ValueError(
                    f"Unsupported mark_input_dim={self.mark_input_dim}. "
                    f"Expected {self.hidden_dim_of_gpt2} or {self.hidden_dim_of_gpt2 * 2}."
                )

        print("mark_input_dim =", self.mark_input_dim)
        print("use_weather =", getattr(self, "use_weather", None))

        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

        # ===== tokenizer / detokenizer =====
        if configs.mlp_hidden_layers == 0:
            print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_gpt2)
            self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.token_len)
        else:
            print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(
                self.token_len,
                self.hidden_dim_of_gpt2,
                configs.mlp_hidden_dim,
                configs.mlp_hidden_layers,
                configs.dropout,
                configs.mlp_activation,
            )
            self.decoder = MLP(
                self.hidden_dim_of_gpt2,
                self.token_len,
                configs.mlp_hidden_dim,
                configs.mlp_hidden_layers,
                configs.dropout,
                configs.mlp_activation,
            )

        # ===== Multi-scale token =====
        self.use_multiscale = getattr(configs, "use_multiscale", False)
        self.ms_fusion = getattr(configs, "ms_fusion", "sum")   # sum / weighted
        self.ms_pattern_pool = getattr(configs, "ms_pattern_pool", 4)

        if self.use_multiscale:
            if self.token_len % self.ms_pattern_pool != 0:
                raise ValueError(
                    f"token_len={self.token_len} must be divisible by "
                    f"ms_pattern_pool={self.ms_pattern_pool}"
                )

            self.pattern_len = self.token_len // self.ms_pattern_pool
            self.rr_len = (self.token_len - 1) + self.token_len

            self.fine_encoder = nn.Linear(self.token_len, self.hidden_dim_of_gpt2)
            self.pattern_encoder = nn.Linear(self.pattern_len, self.hidden_dim_of_gpt2)
            self.rr_encoder = nn.Linear(self.rr_len, self.hidden_dim_of_gpt2)

            if self.ms_fusion == "weighted":
                # 和你当前 Llama 版保持一致：直接标量可学习
                self.alpha_fine = nn.Parameter(torch.tensor(0.0))
                self.alpha_pattern = nn.Parameter(torch.tensor(0.0))
                self.alpha_rr = nn.Parameter(torch.tensor(0.0))
        self.ms_components = getattr(configs, "ms_components", "fine,pattern,rr")
        self.ms_components = [x.strip() for x in self.ms_components.split(",") if x.strip()]
        print("[MODEL INIT] ms_components =", self.ms_components)
        if self.use_multiscale:
            if "fine" not in self.ms_components:
                raise ValueError("ms_components must include 'fine'. Recommended: fine / fine,pattern / fine,rr / fine,pattern,rr")

        print("[MODEL INIT] use_multiscale =", self.use_multiscale)
        if self.use_multiscale:
            print("[MODEL INIT] ms_fusion =", self.ms_fusion)
            print("[MODEL INIT] pattern_len =", self.pattern_len)

        # ===== calendar prefix =====
        self.use_prefix = getattr(configs, "use_prefix", False)
        self.prefix_calendar_dim = getattr(configs, "prefix_calendar_dim", 18)

        if self.use_prefix:
            self.calendar_prefix_proj = nn.Linear(self.prefix_calendar_dim, self.hidden_dim_of_gpt2)
            self.calendar_prefix_norm = nn.LayerNorm(self.hidden_dim_of_gpt2)
        self.use_social_prefix = getattr(configs, "use_social_prefix", False)
        self.prefix_social_dim = getattr(configs, "prefix_social_dim", 6)
        if self.use_social_prefix:
            self.social_prefix_proj = nn.Linear(self.prefix_social_dim, self.hidden_dim_of_gpt2)
            self.social_prefix_norm = nn.LayerNorm(self.hidden_dim_of_gpt2)
        print("[MODEL INIT] use_prefix =", self.use_prefix)
    def build_social_prefix_embeds(self, prefix_social, bs, n_vars):
        if prefix_social is None:
            return None

        soc = self.social_prefix_proj(prefix_social)
        soc = self.social_prefix_norm(soc)
        soc = soc.unsqueeze(1).repeat(1, n_vars, 1)
        soc = soc.reshape(bs * n_vars, 1, self.hidden_dim_of_gpt2)
        return soc
    def build_multiscale_embeds(self, fold_out):
        """
        fold_out: [B*C, token_num, token_len]
        return:   [B*C, token_num, hidden_dim]
        """
        fine = fold_out  # [BC, T, token_len]

        p = self.ms_pattern_pool

        # ===== pattern prior: 用历史日块平均 profile，而不是当天自身粗采样 =====
        pattern_prior = fold_out.mean(dim=1, keepdim=True)                # [BC, 1, token_len]
        pattern_prior = pattern_prior.repeat(1, fold_out.shape[1], 1)     # [BC, T, token_len]

        pattern = pattern_prior.reshape(
            fold_out.shape[0], fold_out.shape[1], self.pattern_len, p
        ).mean(dim=-1)                                                    # [BC, T, pattern_len]

        # ramp: 平滑后的一阶差分
        x_pad = torch.nn.functional.pad(fold_out, (1, 1), mode='replicate')
        x_smooth = (x_pad[..., :-2] + x_pad[..., 1:-1] + x_pad[..., 2:]) / 3.0
        ramp = x_smooth[..., 1:] - x_smooth[..., :-1]                     # [BC, T, token_len-1]

        # residual: 原始 - pattern上采样
        pattern_up = pattern.repeat_interleave(p, dim=-1)                 # [BC, T, token_len]
        residual = fold_out - pattern_up                                  # [BC, T, token_len]

        rr = torch.cat([ramp, residual], dim=-1)                          # [BC, T, rr_len]

        z_fine = self.fine_encoder(fine)
        z_pattern = self.pattern_encoder(pattern)
        z_rr = self.rr_encoder(rr)

        use_fine = "fine" in self.ms_components
        use_pattern = "pattern" in self.ms_components
        use_rr = "rr" in self.ms_components

        if not use_fine:
            raise ValueError("fine branch must be enabled in ms_components")

        if self.ms_fusion == "sum":
            z = 0.0
            if use_fine:
                z = z + z_fine
            if use_pattern:
                z = z + z_pattern
            if use_rr:
                z = z + z_rr

        elif self.ms_fusion == "weighted":
            z = 0.0
            if use_fine:
                z = z + self.alpha_fine * z_fine
            if use_pattern:
                z = z + self.alpha_pattern * z_pattern
            if use_rr:
                z = z + self.alpha_rr * z_rr

        else:
            raise ValueError(f"Unsupported ms_fusion={self.ms_fusion}")

        return z

    def build_prefix_embeds(self, prefix_calendar, bs, n_vars):
        """
        prefix_calendar: [B, 18]
        return: [B*C, 1, H]
        """
        if prefix_calendar is None:
            return None

        cal = self.calendar_prefix_proj(prefix_calendar)          # [B, H]
        cal = self.calendar_prefix_norm(cal)
        cal = cal.unsqueeze(1).repeat(1, n_vars, 1)              # [B, C, H]
        cal = cal.reshape(bs * n_vars, 1, self.hidden_dim_of_gpt2)
        return cal

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, prefix_calendar=None, prefix_social=None):
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

            times_embeds = times_embeds / (times_embeds.norm(dim=2, keepdim=True) + 1e-8)

            if not getattr(self, "use_weather", False):
                # only time.pt or single fused mark
                time_mark = self.time_proj(x_mark_enc)
                time_mark = time_mark / (time_mark.norm(dim=2, keepdim=True) + 1e-8)
                fused_mark = self.mark_fuse(time_mark)

            else:
                # time.pt + weather.pt
                time_mark = x_mark_enc[..., :self.hidden_dim_of_gpt2]
                weather_mark = x_mark_enc[..., self.hidden_dim_of_gpt2:]

                time_feat = self.time_proj(time_mark)
                weather_feat = self.weather_proj(weather_mark)

                time_feat = self.time_norm(time_feat)
                weather_feat = self.weather_norm(weather_feat)

                time_feat = time_feat / (time_feat.norm(dim=2, keepdim=True) + 1e-8)
                weather_feat = weather_feat / (weather_feat.norm(dim=2, keepdim=True) + 1e-8)

                gate_in = torch.cat([times_embeds, time_feat, weather_feat], dim=-1)
                gates = self.gate_net(gate_in)
                g_base, g_time, g_weather = torch.chunk(gates, 3, dim=-1)

                gated_sum = (
                    g_base * times_embeds +
                    g_time * time_feat +
                    g_weather * weather_feat
                )

                interaction = time_feat * weather_feat

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

        # ===== calendar prefix =====
        prefix_list = []

        if self.use_prefix:
            cal_prefix = self.build_prefix_embeds(prefix_calendar, bs, n_vars)
            if cal_prefix is not None:
                prefix_list.append(cal_prefix)

        if self.use_social_prefix:
            soc_prefix = self.build_social_prefix_embeds(prefix_social, bs, n_vars)
            if soc_prefix is not None:
                prefix_list.append(soc_prefix)

        prefix_embeds = None
        if len(prefix_list) > 0:
            prefix_embeds = torch.cat(prefix_list, dim=1)
            prefix_embeds = prefix_embeds.to(device=times_embeds.device, dtype=times_embeds.dtype)
            times_embeds = torch.cat([prefix_embeds, times_embeds], dim=1)
        # GPT2 forward
        outputs = self.gpt2(inputs_embeds=times_embeds).last_hidden_state

        # 如果加了 prefix，decoder 前裁掉
        if prefix_embeds is not None:
            prefix_len = prefix_embeds.shape[1]
            outputs = outputs[:, prefix_len:, :]

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

    def forward(
        self, x_enc, x_mark_enc, x_dec, x_mark_dec,
        prefix_calendar=None, prefix_social=None
    ):
        return self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec,
            prefix_calendar=prefix_calendar,
            prefix_social=prefix_social
        )