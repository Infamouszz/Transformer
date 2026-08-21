import regex as re
from datasets import load_dataset
import json
import TrainingAlgorithm
dataset = load_dataset("jvanz/portuguese_wikipedia_sentences", split="train")
sample = dataset.shuffle().select(range(60_000))

pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
compiled_pattern = re.compile(pattern)

text = [item["text"] for item in sample]

encoded_splitted_text = [
    list(chunk.encode('utf-8'))
    for doc in text
    for chunk in compiled_pattern.findall(doc)
]

tokenizer = TrainingAlgorithm.Tokenizer(encoded_splitted_text)

tokens1, merges = tokenizer.train(merges=10000, start_token=256)
after = sum(len(sublista) for sublista in tokens1)
merge_list = [list(pair) for pair in merges.keys()]

with open("merges.json", "w", encoding="utf-8") as f:
    json.dump(merge_list, f, indent = 2)

print("Merge.json saved successfully!")