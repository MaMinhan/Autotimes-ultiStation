import argparse
import os
import datetime
import pandas as pd
import holidays
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from tqdm import tqdm


class Dataset_Preprocess(Dataset):
    """
    Build prompts from a unique, sorted datetime axis.
    Each time point -> one prompt: from t to t + (token_len-1)*freq
    Add holiday semantics into the prompt.
    """
    def __init__(
        self,
        root_path,
        size=None,
        data_path=None,
        freq_minutes=15,
        holiday_country="AU",
        holiday_subdiv="NSW",
    ):
        assert size is not None
        self.seq_len, self.label_len, self.pred_len = size
        self.token_len = self.seq_len - self.label_len
        self.freq_minutes = int(freq_minutes)

        fp = os.path.join(root_path, data_path) if root_path else data_path
        df = pd.read_csv(fp, usecols=["datetime"])
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])

        # 全局唯一时间轴
        dt = df["datetime"].drop_duplicates().sort_values().reset_index(drop=True)
        self.dt = dt.tolist()

        # 节假日日历
        self.holiday_calendar = holidays.country_holidays(
            holiday_country,
            subdiv=holiday_subdiv
        )

    def __len__(self):
        return len(self.dt)

    def __getitem__(self, idx):
        start = self.dt[idx]
        end = start + datetime.timedelta(
            minutes=self.freq_minutes * (self.token_len - 1)
        )

        day = start.date()
        is_holiday = day in self.holiday_calendar
        holiday_name = self.holiday_calendar.get(day, "")

        holiday_text = (
            f"It is a public holiday ({holiday_name})."
            if is_holiday
            else "It is not a public holiday."
        )

        return (
            f"{holiday_text}"
        )


@torch.no_grad()
def embed_batch(model, tokenizer, texts, device, max_length=128):
    # 显式在文本末尾拼 EOS
    texts = [t + " " + tokenizer.eos_token for t in texts]

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=False,   # EOS 已手动拼接
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(**enc)
    last_hidden = out.last_hidden_state   # [B, L, D]

    # 最后一个有效 token，即我们手动拼进去的 EOS
    eos_idx = enc["attention_mask"].sum(dim=1) - 1
    batch_idx = torch.arange(last_hidden.size(0), device=device)
    emb = last_hidden[batch_idx, eos_idx, :]   # [B, D]

    return emb.float().cpu()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--llm_ckp_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--freq_minutes", type=int, default=15)
    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--label_len", type=int, default=576)
    parser.add_argument("--pred_len", type=int, default=96)

    parser.add_argument("--holiday_country", type=str, default="AU")
    parser.add_argument("--holiday_subdiv", type=str, default="NSW")

    args = parser.parse_args()

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
        holiday_country=args.holiday_country,
        holiday_subdiv=args.holiday_subdiv,
    )

    print("Total prompts:", len(dataset))
    if len(dataset) > 0:
        print("Sample prompt 0:")
        print(dataset[0])
    if len(dataset) > 100:
        print("Sample prompt 100:")
        print(dataset[100])

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_embeddings = []
    for batch in tqdm(loader, desc="Embedding"):
        emb = embed_batch(
            model=model,
            tokenizer=tokenizer,
            texts=batch,
            device=device,
            max_length=args.max_length,
        )
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

'''python /root/autotimes/embedding构建/holiday/build_time_pt_from_long_formate_csv_add_holiday_eos.py \
  --gpu 0 \
  --llm_ckp_dir /root/autodl-tmp/hf_models/gpt2 \
  --data_path /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/merged_include_id_filled.csv \
  --save_path /root/autodl-tmp/datasets/SelfMadeAusgridData/time_embedding/only_holiday_embedding_byGPT2_token96.pt \
  --batch_size 64 \
  --freq_minutes 15 \
  --max_length 128 \
  --seq_len 672 \
  --label_len 576 \
  --pred_len 96 \
  --holiday_country AU \
  --holiday_subdiv NSW'''