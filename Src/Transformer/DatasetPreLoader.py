import torch

class DataLoaderPackerLocal:
    def __init__(self, base_path, max_parts, seq_len, batch_size):
        self.file_paths = [f"{base_path}/dataset_parte_{i}.pt" for i in range(max_parts)]
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.data_cache = []

        for i in range(max_parts):
            fp = f"{base_path}/dataset_parte_{i}.pt"
            try:
                data = torch.load(fp, map_location="cpu", weights_only=True)
                self.data_cache.append(data)
            except FileNotFoundError:
                print(f"Arquivo {fp} não encontrado, pulando...")


    def load(self):
        chunk_size = self.seq_len + 1

        for data in self.data_cache:
            total_chunks = len(data) // chunk_size
            data_sliced = data[:total_chunks * chunk_size]
            data_2d = data_sliced.view(total_chunks, chunk_size)

            X_all = data_2d[:, :-1]
            Y_all = data_2d[:, 1:]

            X_all = X_all.pin_memory()
            Y_all = Y_all.pin_memory()

            for i in range(0, total_chunks, self.batch_size):
                batch_X = X_all[i: i + self.batch_size]
                batch_Y = Y_all[i: i + self.batch_size]

                if len(batch_X) == self.batch_size:
                    yield batch_X, batch_Y

    def get_max_steps(self):
        chunk_size = self.seq_len + 1
        total_steps = 0
        for data in self.data_cache:
            total_chunks = len(data) // chunk_size
            total_steps += total_chunks // self.batch_size
        return total_steps