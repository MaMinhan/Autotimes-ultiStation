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
                # only time.pt
                self.use_weather = False
                self.time_proj = nn.Linear(self.hidden_dim_of_gpt2, self.hidden_dim_of_gpt2)
                self.mark_fuse = nn.Identity()

            elif self.mark_input_dim == self.hidden_dim_of_gpt2 * 2:
                # time.pt + weather.pt
                self.use_weather = True
                self.time_proj = nn.Linear(self.hidden_dim_of_gpt2, self.hidden_dim_of_gpt2)
                self.weather_proj = nn.Linear(self.hidden_dim_of_gpt2, self.hidden_dim_of_gpt2)
                self.mark_fuse = nn.Linear(self.hidden_dim_of_gpt2 * 2, self.hidden_dim_of_gpt2)

            else:
                raise ValueError(
                    f"Unsupported mark_input_dim={self.mark_input_dim}. "
                    f"Expected 768 or 1536."
                )

        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

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

        # [B*C, token_num, 768]
        times_embeds = self.encoder(fold_out)

        if self.mix:
            if x_mark_enc is None:
                raise ValueError("mix_embeds=True but x_mark_enc is None")

            #print("[MODEL DEBUG] entering mix branch")

            times_embeds = times_embeds / (times_embeds.norm(dim=2, keepdim=True) + 1e-8)

            if not getattr(self, "use_weather", False):
                time_mark = self.time_proj(x_mark_enc)
                time_mark = time_mark / (time_mark.norm(dim=2, keepdim=True) + 1e-8)
                fused_mark = self.mark_fuse(time_mark)

            else:
                time_mark = x_mark_enc[..., :self.hidden_dim_of_gpt2]
                weather_mark = x_mark_enc[..., self.hidden_dim_of_gpt2:]

                time_mark = self.time_proj(time_mark)
                weather_mark = self.weather_proj(weather_mark)

                time_mark = time_mark / (time_mark.norm(dim=2, keepdim=True) + 1e-8)
                weather_mark = weather_mark / (weather_mark.norm(dim=2, keepdim=True) + 1e-8)

                fused_mark = torch.cat([time_mark, weather_mark], dim=-1)
                fused_mark = self.mark_fuse(fused_mark)
                fused_mark = fused_mark / (fused_mark.norm(dim=2, keepdim=True) + 1e-8)

            if times_embeds.shape != fused_mark.shape:
                raise RuntimeError(
                    f"shape mismatch after fusion: times_embeds={times_embeds.shape}, "
                    f"fused_mark={fused_mark.shape}"
                )

            times_embeds = times_embeds + self.add_scale * fused_mark

        outputs = self.gpt2(inputs_embeds=times_embeds).last_hidden_state

        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(bs, n_vars, -1)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)