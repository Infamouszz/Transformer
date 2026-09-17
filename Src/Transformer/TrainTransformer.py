import torch
from Src.Transformer.DatasetPreLoader import DataLoaderPackerLocal
from Src.Transformer.Optimizer.AdamOpimizer import AdamOptimizer
from Src.Transformer.Settings.Config import DEVICE
from Src.Transformer.Transformer import Transformer

transformer = Transformer(epochs=4, batch_size=16, d_model=512, vocab_size=10263, causal_mask_size=2048, max_seq_len=512, num_blocks=6)
transformer.to(DEVICE)
dataloader = DataLoaderPackerLocal(base_path=f"/kaggle/working/DatasetAiSFT_FINAL", max_parts=0, seq_len=512, batch_size=1)
total_steps = dataloader.get_max_steps() * transformer.epochs
optimizer = AdamOptimizer(parameters=transformer.get_params(), alpha=3e-4, total_steps=total_steps)

steps = 0

transformer.load_params(r"/kaggle/working/parameters_sft.pt")
print("Parameters loaded")
print("Training started")
for e in range(transformer.epochs):
    with torch.no_grad():
        for X, Y in dataloader.load():
            print("Batch")
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)

            Y_flat = Y.reshape(-1)

            logits = transformer.forward(X)
            logits_flat = logits.view(-1, logits.size(-1))
            loss = torch.nn.functional.cross_entropy(logits_flat, Y_flat)

            probs = torch.softmax(logits_flat, dim=-1)
            dZ_flat = probs.clone()

            dZ_flat[torch.arange(Y_flat.size(0)), Y_flat] -= 1.0

            dZ = dZ_flat.view(logits.shape)

            transformer.backward(dZ)

            optimizer.clip_grad_norm_(transformer.get_params_grads(), 1.0)

            optimizer.update(transformer.get_params_grads())

            optimizer.zero_grad(transformer.get_params_grads())

            steps += 1

            print(f"Loss: {loss.item():} | Steps: {steps} | Total steps: {total_steps} | Epochs: {e} | Learning Rate: {optimizer.cosine_decay_warmup(steps, total_steps)}")

print("Training finished")
transformer.save_params(r"/kaggle/working/parameters_sft_final.pt")
print("Parameters saved")
