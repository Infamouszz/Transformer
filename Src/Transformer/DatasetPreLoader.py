import torch
import os
from dotenv import load_dotenv, find_dotenv

from Src.Transformer.Settings.Config import DEVICE
from Src.Transformer.Tokenizer import Tokenizer
from huggingface_hub import login
load_dotenv(find_dotenv())
login(token=os.getenv("HF_TOKEN"))
from datasets import load_dataset

dataset = load_dataset(
    "Polygl0t/gigaverbo-v2",
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


class DataLoaderPackerLocal:
    def __init__(self, base_path, max_parts, seq_len, batch_size):
        self.file_paths = [f"{base_path}/dataset_parte_{i}.pt" for i in range(max_parts)]
        self.seq_len = seq_len
        self.batch_size = batch_size

    def load(self):
        chunk_size = self.seq_len + 1

        for fp in self.file_paths:
            try:
                data = torch.load(fp, map_location="cpu", weights_only=True)
            except FileNotFoundError:
                print(f"Arquivo {fp} não encontrado, pulando...")
                continue

            total_chunks = len(data) // chunk_size
            data = data[:total_chunks * chunk_size]

            data_2d = data.view(total_chunks, chunk_size)

            X_all = data_2d[:, :-1]
            Y_all = data_2d[:, 1:]

            for i in range(0, total_chunks, self.batch_size):
                batch_X = X_all[i: i + self.batch_size]
                batch_Y = Y_all[i: i + self.batch_size]

                if len(batch_X) == self.batch_size:
                    yield batch_X, batch_Y

            del data, data_2d, X_all, Y_all