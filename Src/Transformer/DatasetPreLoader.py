import torch

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