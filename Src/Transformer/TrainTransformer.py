import torch
from Src.Transformer.DatasetPreLoader import DataLoaderPackerLocal
from Src.Transformer.Optimizer.AdamOpimizer import AdamOptimizer
from Src.Transformer.Settings.Config import DEVICE
from Src.Transformer.Transformer import Transformer
from tqdm import tqdm

transformer = Transformer(epochs=10, batch_size=16, d_model=512, vocab_size=10259, causal_mask_size=2048, max_seq_len=256, num_blocks=6)
transformer.to(DEVICE)
optimizer = AdamOptimizer(parameters=transformer.get_params(), alpha=1e-3)
dataloader = DataLoaderPackerLocal(base_path=f"/kaggle/input/datasets/infamouszz/datasetai/DatasetAi", max_parts=10, seq_len=256, batch_size=16)

steps = 0
with torch.no_grad():
    for X, Y in tqdm(dataloader.load(), desc="Training..."):
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)
        Y_flat = Y.view(-1)

        logits = transformer.forward(X)
        logits_flat = logits.view(-1, logits.size(-1))
        loss = torch.nn.functional.cross_entropy(logits_flat, Y_flat)

        predicts = torch.argmax(logits_flat, dim=-1)
        total_tokens = Y_flat.size(0)

        probs = torch.softmax(logits_flat, dim=-1)
        dZ_flat = probs.clone()

        dZ_flat[torch.arange(Y_flat.size(0)), Y_flat] -= 1.0

        dZ = dZ_flat.view(logits.shape)

        transformer.backward(dZ)

        optimizer.update(transformer.get_params_grads())

        optimizer.zero_grad(transformer.get_params_grads())

        steps += 1

        if steps % 100 == 0:
            print(f"Loss: {loss.item():}")

print("Training finished")
transformer.save_params(r"E:/ModelAi/parameters.pt")
print("Parameters saved")
