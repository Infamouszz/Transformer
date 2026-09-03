import torch
from Src.Transformer.DatasetPreLoader import DataLoaderPacker
from Src.Transformer.Optimizer.AdamOpimizer import AdamOptimizer
from Src.Transformer.Settings.Config import DEVICE
from Src.Transformer.Transformer import Transformer
from tqdm import tqdm

transformer = Transformer(epochs=10, batch_size=32, d_model=512, vocab_size=10259, causal_mask_size=2048, max_seq_len=512)
optimizer = AdamOptimizer(parameters=transformer.get_params(), alpha=3e-4)
dataloader = DataLoaderPacker(seq_len=512, batch_size=32)

steps = 0
max_steps = 50_000
for X, Y in tqdm(dataloader.load(), desc="Training..."):
    X = X.to(DEVICE)
    Y = Y.to(DEVICE)

    logits = transformer.forward(X)
    logits_flat = logits.view(-1, logits.size(-1))
    Y_flat = Y.view(-1)

    loss = torch.nn.functional.cross_entropy(logits_flat, Y_flat)

    predicts = torch.argmax(logits_flat, dim=-1)
    hits = (predicts == Y_flat).sum().item()
    total_tokens = Y_flat.size(0)
    accuracy = (hits / total_tokens) * 100

    probs = torch.softmax(logits_flat, dim=-1)
    dZ_flat = probs.clone()

    dZ_flat[torch.arange(Y_flat.size(0)), Y_flat] -= 1.0

    dZ_flat = dZ_flat / Y_flat.size(0)

    dZ = dZ_flat.view(logits.shape)

    transformer.backward(dZ)

    optimizer.update(transformer.get_params_grads())

    optimizer.zero_grad(transformer.get_params_grads())

    steps += 1

    print(f"Loss: {loss.item():} | Acurácia: {accuracy}%")

    if steps % 100 == 0:
        print(steps)

    if steps >= max_steps:
        break

