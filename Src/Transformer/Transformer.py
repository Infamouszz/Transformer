import torch
from Src.Transformer.Embedding import Embedding
from Src.Transformer.Embedding.Unembedding import Unembedding
from Src.Transformer.Normalization import LayerNormalization
from Src.Transformer.Settings.Config import DEVICE
from Src.Transformer.TransformerBlock import TransformerBlock


class Transformer:
    def __init__(self, epochs, batch_size, d_model, vocab_size, causal_mask_size, max_seq_len, num_blocks):
        self.epochs = epochs
        self.batch_size = batch_size
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.emb = Embedding.Embedding(self.d_model)
        self.une = Unembedding(self.vocab_size)
        self.embedding = self.emb.initialize_embedding_HE(self.vocab_size)

        pos_emb = self.emb.initialize_sinusoidal_positional_embedding(self.max_seq_len)
        if pos_emb.ndim == 2:
            pos_emb = pos_emb.unsqueeze(0)
        self.positional_embedding = pos_emb

        self.W_unembedding, self.b_unembedding = self.une.initialize_unembedding_HE(self.d_model)

        self.norm3 = LayerNormalization.LayerNorm(self.d_model)

        self.blocks = []
        for _ in range(num_blocks):
            self.blocks.append(TransformerBlock.TransformerBlock(d_model, causal_mask_size, max_seq_len))

        self.cache = {}

    def forward(self, X_tokens):
        current_batch_size, seq_len = X_tokens.shape

        pos_emb = self.positional_embedding[:, :seq_len, :]

        X_embedding = self.emb.forward(X_tokens, self.embedding, pos_emb)

        block_output = X_embedding
        for block in self.blocks:
            block_output = block.forward(block_output)

        final_X_norm = self.norm3.forward(block_output)

        logits_probs = self.une.forward(final_X_norm, self.W_unembedding, self.b_unembedding)

        self.cache = {
            'X_tokens': X_tokens,
            'final_X_norm': final_X_norm,
        }

        return logits_probs

    def backward(self, dZ):
        dX_norm_final, self.dW_une, self.db_une = self.une.backward(dZ, self.cache['final_X_norm'], self.W_unembedding)

        dX_block, self.dgamma_norm3, self.dbeta_norm3 = self.norm3.backward(dX_norm_final)

        for block in reversed(self.blocks):
            dX_block = block.backward(dX_block)

        dX_embedded = dX_block / torch.sqrt(self.d_model)

        self.dW_embed = torch.zeros_like(self.embedding)

        tokens_flat = self.cache['X_tokens'].reshape(-1).to(device=DEVICE)
        grad_flat = dX_embedded.reshape(-1, self.d_model)

        self.dW_embed.index_add_(0, tokens_flat, grad_flat)


    def get_params(self):
        params = [self.embedding]
        for block in self.blocks:
            params.extend([
                block.norm1.gamma, block.norm1.beta,
                block.Wq, block.Wk, block.Wv, block.Wo,
                block.norm2.gamma, block.norm2.beta,
                block.ffn.W1, block.ffn.b1, block.ffn.W2, block.ffn.b2
            ])
        params.extend([
            self.norm3.gamma, self.norm3.beta,
            self.W_unembedding, self.b_unembedding
        ])
        return params

    def get_params_grads(self):
        grads = [(self.embedding, self.dW_embed)]
        for block in self.blocks:
            grads.extend([
                (block.norm1.gamma, block.dgamma_norm1),
                (block.norm1.beta, block.dbeta_norm1),
                (block.Wq, block.dWq),
                (block.Wk, block.dWk),
                (block.Wv, block.dWv),
                (block.Wo, block.dWo),
                (block.norm2.gamma, block.dgamma_norm2),
                (block.norm2.beta, block.dbeta_norm2),
                (block.ffn.W1, block.ffn.dW1),
                (block.ffn.b1, block.ffn.db1),
                (block.ffn.W2, block.ffn.dW2),
                (block.ffn.b2, block.ffn.db2)
            ])
        grads.extend([
            (self.norm3.gamma, self.dgamma_norm3),
            (self.norm3.beta, self.dbeta_norm3),
            (self.W_unembedding, self.dW_une),
            (self.b_unembedding, self.db_une)
        ])
        return grads

    def to(self, device):
        self.embedding = self.embedding.to(device)
        self.positional_embedding = self.positional_embedding.to(device)
        self.W_unembedding = self.W_unembedding.to(device)
        self.b_unembedding = self.b_unembedding.to(device)
        self.norm3.gamma = self.norm3.gamma.to(device)
        self.norm3.beta = self.norm3.beta.to(device)

        for block in self.blocks:
            block.Wq = block.Wq.to(device)
            block.Wk = block.Wk.to(device)
            block.Wv = block.Wv.to(device)
            block.Wo = block.Wo.to(device)
            block.norm1.gamma = block.norm1.gamma.to(device)
            block.norm1.beta = block.norm1.beta.to(device)
            block.norm2.gamma = block.norm2.gamma.to(device)
            block.norm2.beta = block.norm2.beta.to(device)
            block.ffn.W1 = block.ffn.W1.to(device)
            block.ffn.b1 = block.ffn.b1.to(device)
            block.ffn.W2 = block.ffn.W2.to(device)
            block.ffn.b2 = block.ffn.b2.to(device)
        return self

    def save_params(self, path):
        state_dict = {
            'embedding': self.embedding,
            'norm3_gamma': self.norm3.gamma,
            'norm3_beta': self.norm3.beta,
            'W_unembedding': self.W_unembedding,
            'b_unembedding': self.b_unembedding
        }
        for i, block in enumerate(self.blocks):
            state_dict[f'block_{i}_Wq'] = block.Wq
            state_dict[f'block_{i}_Wk'] = block.Wk
            state_dict[f'block_{i}_Wv'] = block.Wv
            state_dict[f'block_{i}_Wo'] = block.Wo
            state_dict[f'block_{i}_norm1_gamma'] = block.norm1.gamma
            state_dict[f'block_{i}_norm1_beta'] = block.norm1.beta
            state_dict[f'block_{i}_norm2_gamma'] = block.norm2.gamma
            state_dict[f'block_{i}_norm2_beta'] = block.norm2.beta
            state_dict[f'block_{i}_ffn_W1'] = block.ffn.W1
            state_dict[f'block_{i}_ffn_b1'] = block.ffn.b1
            state_dict[f'block_{i}_ffn_W2'] = block.ffn.W2
            state_dict[f'block_{i}_ffn_b2'] = block.ffn.b2
        torch.save(state_dict, path)

    def load_params(self, path):
        state_dict = torch.load(path, map_location=DEVICE, weights_only=True)

        self.embedding.data.copy_(state_dict['embedding'])
        self.norm3.gamma.data.copy_(state_dict['norm3_gamma'])
        self.norm3.beta.data.copy_(state_dict['norm3_beta'])
        self.W_unembedding.data.copy_(state_dict['W_unembedding'])
        self.b_unembedding.data.copy_(state_dict['b_unembedding'])

        for i, block in enumerate(self.blocks):
            block.Wq.data.copy_(state_dict[f'block_{i}_Wq'])
            block.Wk.data.copy_(state_dict[f'block_{i}_Wk'])
            block.Wv.data.copy_(state_dict[f'block_{i}_Wv'])
            block.Wo.data.copy_(state_dict[f'block_{i}_Wo'])
            block.norm1.gamma.data.copy_(state_dict[f'block_{i}_norm1_gamma'])
            block.norm1.beta.data.copy_(state_dict[f'block_{i}_norm1_beta'])
            block.norm2.gamma.data.copy_(state_dict[f'block_{i}_norm2_gamma'])
            block.norm2.beta.data.copy_(state_dict[f'block_{i}_norm2_beta'])
            block.ffn.W1.data.copy_(state_dict[f'block_{i}_ffn_W1'])
            block.ffn.b1.data.copy_(state_dict[f'block_{i}_ffn_b1'])
            block.ffn.W2.data.copy_(state_dict[f'block_{i}_ffn_W2'])
            block.ffn.b2.data.copy_(state_dict[f'block_{i}_ffn_b2'])




