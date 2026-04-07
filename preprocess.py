import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from data_provider.data_loader import Dataset_Preprocess
from tqdm import tqdm


@torch.no_grad()
def embed_batch(model, tokenizer, texts, device):
    # 显式在文本末尾加 EOS
    texts = [t + tokenizer.eos_token for t in texts]

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
        add_special_tokens=False,   # GPT2 不靠这个自动加 EOS，我们已经手动拼了
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(**enc)
    last_hidden = out.last_hidden_state   # [B, L, D]

    # 因为我们手动在最后拼了 eos_token，所以最后一个有效 token 就是 EOS
    eos_idx = enc["attention_mask"].sum(dim=1) - 1   # [B]
    batch_idx = torch.arange(last_hidden.size(0), device=device)
    emb = last_hidden[batch_idx, eos_idx, :]         # [B, D]

    return emb.float().cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--llm_ckp_dir", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--freq_minutes", type=int, default=15)
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.llm_ckp_dir)

    if tokenizer.eos_token is None:
        raise ValueError("GPT2 tokenizer has no eos_token. Please check llm_ckp_dir.")

    # GPT2 没有 pad_token，通常用 eos_token 代替
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2Model.from_pretrained(args.llm_ckp_dir).to(device)
    model.eval()

    dataset = Dataset_Preprocess(
        root_path="",
        data_path=args.data_path,
        size=[672, 576, 96],
        freq_minutes=args.freq_minutes,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    all_embeddings = []

    print(f"Total samples: {len(dataset)}")
    print(f"Using eos token: {repr(tokenizer.eos_token)}")

    for batch in tqdm(loader, desc="Embedding"):
        emb = embed_batch(model, tokenizer, batch, device)
        all_embeddings.append(emb)

    result = torch.cat(all_embeddings, dim=0)   # [T, 768]

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(result, args.save_path)
    print("Saved embedding:", args.save_path)
    print("Shape:", result.shape)


if __name__ == "__main__":
    main()