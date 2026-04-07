import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from data_provider.data_loader import Dataset_Preprocess
from tqdm import tqdm
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom

@torch.no_grad()
def embed_batch(model, tokenizer, texts, device):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
    return emb.float().cpu()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--llm_ckp_dir", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--freq_minutes", type=int, default=15)
    p.add_argument("--include_station", action="store_true", default=True)
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.llm_ckp_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(args.llm_ckp_dir).to(device)
    model.eval()

    dataset = Dataset_Preprocess(
        root_path="",
        data_path=args.data_path,
        size=[672, 576, 96],
        freq_minutes=args.freq_minutes,
    )
    print("DEBUG data_path:", args.data_path)
    print("DEBUG dataset_len:", len(dataset))
    import pandas as pd
    df = pd.read_csv(args.data_path, usecols=["datetime","station","target"])
    print("DEBUG csv_len:", len(df))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_embeddings = []
    print(f"Total samples: {len(dataset)}")

    for batch in tqdm(loader, desc="Embedding"):
        emb = embed_batch(model, tokenizer, batch, device)
        all_embeddings.append(emb)

    result = torch.cat(all_embeddings, dim=0)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(result, args.save_path)

    print("Saved embedding:", args.save_path)
    print("Shape:", result.shape)

if __name__ == "__main__":
    main()
