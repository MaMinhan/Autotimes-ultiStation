import argparse
import os
import datetime
import pandas as pd
import holidays
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm


class Dataset_Preprocess(Dataset):
    """
    Build prompts from a unique, sorted datetime axis.
    Each time point -> one prompt: from t to t + (token_len-1)*freq
    Add holiday semantics into the prompt.

    这里保持你原来的holiday语义逻辑不变。
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

        # 如果你之后想把时间范围也放回prompt，可以改成：
        return (
            f"{holiday_text} "
            f"This is the series from {start:%Y/%m/%d %H:%M:%S} "
            f"to {end:%Y/%m/%d %H:%M:%S}."
        )

        return f"{holiday_text}"


@torch.no_grad()
def embed_batch(model, tokenizer, texts, device, max_length=128):
    """
    严格按 Preprocess_Llama.py 的方式：
    1. tokenizer
    2. input_ids -> input_embeddings
    3. llama.model(inputs_embeds=...)
    4. 取最后一个位置 [:, -1, :]
    """
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    input_ids = enc["input_ids"].to(device)

    # 对齐 Preprocess_Llama.py：先把 token id 变成 embedding
    input_embeddings = model.get_input_embeddings()(input_ids)

    # 对齐 Preprocess_Llama.py：直接喂 inputs_embeds
    text_outputs = model.model(inputs_embeds=input_embeddings)[0]   # [B, L, H]

    # 对齐 Preprocess_Llama.py：直接取最后一个位置
    emb = text_outputs[:, -1, :]   # [B, H]

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

    result = torch.cat(all_embeddings, dim=0)

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(result, args.save_path)

    print("Saved embedding:", args.save_path)
    print("Shape:", result.shape)


if __name__ == "__main__":
    main()


"""
python /root/autotimes/embedding构建/holiday/Llama_build_holiday_pt_eos.py \
  --gpu 0 \
  --llm_ckp_dir /root/autodl-tmp/hf_models/llama \
  --data_path /root/autodl-tmp/datasets/SelfMadeAusgridData/electricity/load_10stations_20240101_20240430.csv \
  --save_path /root/autodl-tmp/datasets/SelfMadeAusgridData/time_embedding/0101_0429/load_10stations_holiday_and_time_embedding_byLlama_token96.pt \
  --batch_size 8 \
  --freq_minutes 15 \
  --max_length 128 \
  --seq_len 672 \
  --label_len 576 \
  --pred_len 96 \
  --holiday_country AU \
  --holiday_subdiv NSW
"""