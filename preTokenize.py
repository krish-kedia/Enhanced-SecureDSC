# pretokenize.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

SEQ_LEN = 20
CHUNK   = 10_000
CACHE   = f"europarl_cache_seqlen{SEQ_LEN}.pt"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print("Loading dataset...")
dataset = load_dataset("Helsinki-NLP/europarl", "en-fr", split="train")
SIZE    = len(dataset)
print(f"Dataset size: {SIZE:,}")

texts = [row["en"] for row in dataset["translation"]]

print(f"Tokenizing {len(texts):,} samples in chunks of {CHUNK}...")
all_ids = []

for i in range(0, len(texts), CHUNK):
    chunk = texts[i : i + CHUNK]
    enc   = tokenizer(
        chunk,
        padding        = "max_length",
        truncation     = True,
        max_length     = SEQ_LEN,
        return_tensors = "pt"
    )
    all_ids.append(enc["input_ids"])

    if i % 100_000 == 0:
        print(f"  {i:>7,} / {len(texts):,}")

all_ids = torch.cat(all_ids, dim=0)
torch.save(all_ids, CACHE)
print(f"\nSaved → {CACHE}  shape: {all_ids.shape}")