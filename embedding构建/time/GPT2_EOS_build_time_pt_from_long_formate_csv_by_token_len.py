import argparse
import os
import datetime
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
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
def embed_batch(model, tokenizer, texts, device, max_length=128):
    texts = [t + " " + tokenizer.eos_token for t in texts]

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(**enc)
    last_hidden = out.last_hidden_state  # [B, L, D]

    eos_idx = enc["attention_mask"].sum(dim=1) - 1
    batch_idx = torch.arange(last_hidden.size(0), device=device)
    emb = last_hidden[batch_idx, eos_idx, :]

    return emb.float().cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--llm_ckp_dir", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--freq_minutes", type=int, default=15)
    p.add_argument("--max_length", type=int, default=128)

    # 保留 size 参数，兼容你现有风格
    p.add_argument("--seq_len", type=int, default=672)
    p.add_argument("--label_len", type=int, default=576)
    p.add_argument("--pred_len", type=int, default=96)

    # 新增：显式 token_len，优先级高于 seq_len-label_len
    p.add_argument("--token_len", type=int, default=None)

    args = p.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.llm_ckp_dir)
    if tokenizer.eos_token is None:
        raise ValueError("GPT2 tokenizer has no eos_token. Please check llm_ckp_dir.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2Model.from_pretrained(args.llm_ckp_dir).to(device)
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
    print(f"Using eos token: {repr(tokenizer.eos_token)}")
    print(f"Effective token_len: {dataset.token_len}")

    for batch in tqdm(loader, desc="Embedding"):
        emb = embed_batch(model, tokenizer, batch, device, max_length=args.max_length)
        all_embeddings.append(emb)

    result = torch.cat(all_embeddings, dim=0)  # [T, 768]

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(result, args.save_path)
    print("Saved embedding:", args.save_path)
    print("Shape:", result.shape)


if __name__ == "__main__":
    main()