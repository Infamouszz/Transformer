import json
import regex as re

class BPETokenizer:
    def __init__(self, merges_file, pattern):
        self.merges_file = merges_file
        self.compiled_pattern = re.compile(pattern)

        with open(merges_file, "r", encoding="utf-8") as f:
            merges_list = json.load(f)

        self.merges = {tuple(pair): 256 + i for i, pair in enumerate(merges_list)}

        self.vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[p0] + self.vocab[p1]

        next_id = 256 + len(merges_list)
        self.special_tokens = {
            "<PAD>": next_id,
            "<BOS>": next_id + 1,
            "<EOS>": next_id + 2,
            "<UNK>": next_id + 3,
        }
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

    def encode(self, text, add_special_tokens=True):
        chunks = self.compiled_pattern.findall(text)
        tokens = []

        for chunk in chunks:
            chunk_bytes = list(chunk.encode("utf-8"))

            while len(chunk_bytes) >= 2:
                stats = {}
                for i in range(len(chunk_bytes) - 1):
                    pair = (chunk_bytes[i], chunk_bytes[i + 1])
                    if pair in self.merges:
                        stats[pair] = self.merges[pair]

                if not stats:
                    break

                best_pair = min(stats, key=stats.get)
                new_id = self.merges[best_pair]

                new_list = []
                i = 0
                while i < len(chunk_bytes):
                    if i < len(chunk_bytes) - 1 and (chunk_bytes[i], chunk_bytes[i + 1]) == best_pair:
                        new_list.append(new_id)
                        i += 2
                    else:
                        new_list.append(chunk_bytes[i])
                        i += 1
                chunk_bytes = new_list

            tokens.extend(chunk_bytes)

        if add_special_tokens:
            tokens = [self.special_tokens["<BOS>"]] + tokens + [self.special_tokens["<EOS>"]]

        return tokens

    def decode(self, tokens):
        raw_bytes = b"".join(
            self.vocab[idx] for idx in tokens if idx in self.vocab
        )
        return raw_bytes.decode("utf-8", errors="replace")