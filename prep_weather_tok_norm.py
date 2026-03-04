import torch

src = "/root/autodl-tmp/datasets/SelfMadeAusgridData/weather_language_V2.pt"
dst = "/root/autodl-tmp/datasets/SelfMadeAusgridData/weather_language_V2_tok24_norm0.1.pt"

token_len = 24
scale = 0.1
eps = 1e-6

w = torch.load(src, map_location="cpu")          # (N, T, D) float16
w = w[:, ::token_len, :].contiguous()            # (N, Ttok, D) 体积大幅降低
w = w.float()                                    # 现在转 float32 才安全（尺寸小很多）

# z-score：按 (N,T) 求每个维度的均值方差
mean = w.mean(dim=(0,1), keepdim=True)           # (1,1,D)
std  = w.std(dim=(0,1), keepdim=True).clamp_min(eps)

w = (w - mean) / std
w = w * scale
w = w.half()                                     # 可选：转回 float16 省空间

torch.save(w, dst)
print("saved:", dst, w.shape, w.dtype, w.min().item(), w.max().item())