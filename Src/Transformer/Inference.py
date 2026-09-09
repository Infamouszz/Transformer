import torch

from Src.Transformer.Settings.Config import DEVICE
from Src.Transformer.Tokenizer import Tokenizer
from Src.Transformer.Transformer import Transformer

path = r"C:\Users\Guilherme\PycharmProjects\DashboardAi\Src\Transformer\Tokenizer\merges.json"
pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

tokenizer_instance = Tokenizer.BPETokenizer(merges_file=path, pattern=pattern)
transformer_instance = Transformer(epochs=1, batch_size=1, d_model=512, vocab_size=10259, causal_mask_size=2048, max_seq_len=512)
transformer_instance.load_params(r"E:/ModelAi/parameters.pt")

def generate_text(transformer, tokens_tensor, max_tokens_generated, eos_id, temperature=0.3, repetition_penalty=1.2):
    with torch.no_grad():
        current_tokens = tokens_tensor.clone()

        for _ in range(max_tokens_generated):
            if current_tokens.size(1) >= transformer.max_seq_len:
                break

            logits = transformer.forward(current_tokens)
            last_token_logits = logits[0, -1, :] / temperature

            for token_id in set(current_tokens[0].tolist()):
                if last_token_logits[token_id] < 0:
                    last_token_logits[token_id] *= repetition_penalty
                else:
                    last_token_logits[token_id] /= repetition_penalty

            probs = torch.softmax(last_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            if next_token_id.item() == eos_id:
                break

            next_token_tensor = next_token_id.unsqueeze(0)
            current_tokens = torch.cat((current_tokens, next_token_tensor), dim=1)

    return current_tokens

text = "Post "
tokens = tokenizer_instance.encode(text)
seq_len = len(tokens)
tensor_tokens = torch.tensor([tokens]).to(DEVICE)

output_tokens = generate_text(transformer_instance, tensor_tokens, max_tokens_generated=200, eos_id=10258)
tokens_list = output_tokens[0].tolist()
decoded_token = tokenizer_instance.decode(tokens_list)

print("Input text: ", text)
print("Output tokens text: ", decoded_token)