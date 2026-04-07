import argparse
import os
import datetime
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm


class Dataset_Preprocess(Dataset):
    """
    AutoTimes-style preprocess dataset (time-only):
    - Build prompts purely from a unique, sorted datetime axis.
    - Each time point -> one prompt: from t to t + (token_len-1)*freq
    """
    def __init__(self, root_path, size=None, data_path=None, freq_minutes=15, token_len=None):
        assert size is not None
        self.seq_len, self.label_len, self.pred_len = size

        if token_len is not None:
            self.token_len = int(token_len)
        else:
            self.token_len = self.seq_len - self.label_len

        self.freq_minutes = int(freq_minutes)

        fp = os.path.join(root_path, data_path) if root_path else data_path
        df = pd.read_csv(fp, usecols=["datetime"])
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])

        dt = df["datetime"].drop_duplicates().sort_values().reset_index(drop=True)
        self.dt = dt.tolist()

    def __len__(self):
        return len(self.dt)

    def __getitem__(self, idx):
        start = self.dt[idx]
        end = start + datetime.timedelta(minutes=self.freq_minutes * (self.token_len - 1))
        return (
            f"This is the series from {start:%Y/%m/%d %H:%M:%S} "
            f"to {end:%Y/%m/%d %H:%M:%S}."
        )


@torch.no_grad()
def embed_batch(model, tokenizer, texts, device):
    """
    严格按 Preprocess_Llama.py 的方式：
    1. tokenizer -> input_ids
    2. get_input_embeddings()
    3. llama.model(inputs_embeds=...)
    4. 取最后一个 token 的 hidden state: [:, -1, :]
    """
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_ids = enc["input_ids"].to(device)

    # 对齐 Preprocess_Llama.py：先转 embedding
    inputs_embeds = model.get_input_embeddings()(input_ids)

    # 对齐 Preprocess_Llama.py：直接喂 inputs_embeds
    text_outputs = model.model(inputs_embeds=inputs_embeds)[0]   # [B, L, H]

    # 对齐 Preprocess_Llama.py：取最后一个位置
    emb = text_outputs[:, -1, :]   # [B, H]

    return emb.float().cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--llm_ckp_dir", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--freq_minutes", type=int, default=15)

    # 保留 size 参数，兼容你现有风格
    p.add_argument("--seq_len", type=int, default=672)
    p.add_argument("--label_len", type=int, default=576)
    p.add_argument("--pred_len", type=int, default=96)

    # 显式 token_len，优先级高于 seq_len-label_len
    p.add_argument("--token_len", type=int, default=None)

    args = p.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = LlamaTokenizer.from_pretrained(args.llm_ckp_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(
        args.llm_ckp_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    model.eval()

    dataset = Dataset_Preprocess(
        root_path="",
        data_path=args.data_path,
        size=[args.seq_len, args.label_len, args.pred_len],
        freq_minutes=args.freq_minutes,
        token_len=args.token_len,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    all_embeddings = []

    print(f"Total samples: {len(dataset)}")
    print(f"Using pad token: {repr(tokenizer.pad_token)}")
    print(f"Using eos token: {repr(tokenizer.eos_token)}")
    print(f"Effective token_len: {dataset.token_len}")

    for batch in tqdm(loader, desc="Embedding"):
        emb = embed_batch(model, tokenizer, batch, device)
        all_embeddings.append(emb)

    result = torch.cat(all_embeddings, dim=0)

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(result, args.save_path)
    print("Saved embedding:", args.save_path)
    print("Shape:", result.shape)


if __name__ == "__main__":
    main()

'''python Llama_build_time_pt_eos.py \
  --gpu 0 \
  --llm_ckp_dir /root/autodl-tmp/hf_models/llama \
  --data_path /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/load_10stations_20240101_20240430.csv \
  --save_path /root/autodl-tmp/datasets/SelfMadeAusgridData/time_embedding/0101_0429/load_10stations_byLlama_time_embedding_T_4096.pt \
  --batch_size 8 \
  --freq_minutes 15 \
  --seq_len 672 \
  --label_len 576 \
  --pred_len 96 \
  --token_len 96'''