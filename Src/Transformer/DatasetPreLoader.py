import torch
import os
from dotenv import load_dotenv, find_dotenv
from Src.Transformer.Tokenizer import Tokenizer
from huggingface_hub import login
load_dotenv(find_dotenv())
login(token=os.getenv("HF_TOKEN"))
from datasets import load_dataset

dataset = load_dataset(
    "TucanoBR/GigaVerbo",
    split="train",
    streaming=True,
    token=os.getenv("HF_TOKEN"),
)
pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

tokenizer = Tokenizer.BPETokenizer(r"C:\Users\Guilherme\PycharmProjects\DashboardAi\Src\Transformer\Tokenizer\merges.json", pattern)

class DataLoaderPacker:
    def __init__(self, seq_len, batch_size):
        self.seq_len = seq_len
        self.batch_size = batch_size

    def load(self):
        buffer = []
        batch_X, batch_Y = [], []

        for sample in dataset["text"]:
            tokens = tokenizer.encode(sample)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]

                batch_X.append(chunk[:-1])
                batch_Y.append(chunk[1:])

                if len(batch_X) == self.batch_size:
                    yield torch.tensor(batch_X, dtype=torch.long), torch.tensor(batch_Y, dtype=torch.long)
                    batch_X, batch_Y = [], []