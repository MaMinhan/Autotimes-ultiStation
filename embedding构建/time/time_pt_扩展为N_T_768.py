import torch

time = torch.load("/root/autodl-tmp/datasets/SelfMadeAusgridData/merged_include_id_filled_by_GPT2.pt")

N = 181  # station数量

time_expand = time.unsqueeze(0).repeat(N, 1, 1)

print(time_expand.shape)

torch.save(time_expand, "/root/autodl-tmp/datasets/SelfMadeAusgridData/time_embedding_byGPT2_expand.pt")