import torch
from Src.Transformer.Embedding import Embedding
from Src.Transformer.Embedding.Unembedding import Unembedding
from Src.Transformer.Mask import CausalMask
from Src.Transformer.MultiHeadAttention import AttentionWeightsInitializer
from Src.Transformer.MultiHeadAttention import MultiHead
from Src.Transformer.MultiHeadAttention import SelfAttention
from Src.Transformer.Normalization import LayerNormalization
from Src.Transformer.ResidualConnection import ResidualConnectionCalculator
from Src.ForwardNeuralNetwork.Src.NeuralNetwork import NeuralNetwork
from Src.Transformer.Settings.Config import DEVICE


class Transformer:
    def __init__(self, epochs, batch_size, d_model, vocab_size, causal_mask_size, max_seq_len):
        self.epochs = epochs
        self.batch_size = batch_size
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.emb = Embedding.Embedding(self.d_model)
        self.une = Unembedding(self.vocab_size)
        self.mha = MultiHead.MultiHeadAttention
        self.mhd = MultiHead.MultiHeadDivisor(self.d_model)
        self.sa = SelfAttention.SelfAttention(self.d_model)
        self.res = ResidualConnectionCalculator.ResidualConnectionCalculator
        self.ffn = NeuralNetwork.NeuralNetwork(self.d_model, self.d_model * 4)
        self.msk = CausalMask.CausalMask
        self.embedding = self.emb.initialize_embedding_HE(self.vocab_size)

        pos_emb = self.emb.initialize_sinusoidal_positional_embedding(self.max_seq_len)
        if pos_emb.ndim == 2:
            pos_emb = pos_emb.unsqueeze(0)
        self.positional_embedding = pos_emb

        self.W_unembedding, self.b_unembedding = self.une.initialize_unembedding_HE(self.d_model)

        self.causal_mask = self.msk.initialize_causal_mask(causal_mask_size)

        self.Wq, self.Wk, self.Wv, self.Wo = AttentionWeightsInitializer.WeightInitializer(self.d_model).init_weights_HE()
        self.norm1 = LayerNormalization.LayerNorm(self.d_model)
        self.norm2 = LayerNormalization.LayerNorm(self.d_model)
        self.norm3 = LayerNormalization.LayerNorm(self.d_model)

        self.cache = {}

    def forward(self, X_tokens):
        current_batch_size, seq_len = X_tokens.shape

        pos_emb = self.positional_embedding[:, :seq_len, :]

        X_embedding = self.emb.forward(X_tokens, self.embedding, pos_emb)

        X_norm1 = self.norm1.forward(X_embedding)
        Q, K, V = self.mha.forward(X_norm1, self.Wq, self.Wk, self.Wv)

        Q_split, K_split, V_split = self.mhd.rearrange(current_batch_size, Q, K, V, num_heads=32)

        causal_mask_sliced = self.causal_mask[:, :, :seq_len, :seq_len]

        attention_out, attn_weights = self.sa.forward(Q_split, K_split, V_split, causal_mask_sliced)
        mha_out = self.mhd.concatenateWo(current_batch_size, attention_out, self.Wo)

        res1 = self.res.calculate(X_embedding, mha_out)
        X_norm2 = self.norm2.forward(res1)

        y_pred_ffn, ffn_hidden_act = self.ffn.forward(X_norm2)

        res2 = self.res.calculate(res1, y_pred_ffn)
        final_X_norm = self.norm3.forward(res2)

        logits_probs = self.une.forward(final_X_norm, self.W_unembedding, self.b_unembedding)

        self.cache = {
            'X_tokens': X_tokens,
            'X_embedding': X_embedding,
            'X_norm1': X_norm1,
            'Q': Q_split, 'K': K_split, 'V': V_split,
            'attn_weights': attn_weights,
            'attention_out': attention_out,
            'mha_out': mha_out,
            'res1': res1,
            'X_norm2': X_norm2,
            'ffn_hidden_act': ffn_hidden_act,
            'y_pred_ffn': y_pred_ffn,
            'res2': res2,
            'final_X_norm': final_X_norm,
            'logits_probs': logits_probs
        }

        return logits_probs

    def backward(self, dZ):
        dX_norm_final, self.dW_une, self.db_une = self.une.backward(dZ, self.cache['final_X_norm'], self.W_unembedding)

        dX_res2, self.dgamma_norm3, self.dbeta_norm3 = self.norm3.backward(dX_norm_final)

        dY_ffn = dX_res2.clone()
        dX_res1_shortcut = dX_res2.clone()

        dX_norm2_ffn = self.ffn.backward(dY_ffn)

        dX_res1_ffn, self.dgamma_norm2, self.dbeta_norm2 = self.norm2.backward(dX_norm2_ffn)

        dX_res1_total = dX_res1_ffn + dX_res1_shortcut

        d_mha_out = dX_res1_total.clone()
        dX_embed_shortcut = dX_res1_total.clone()

        dX_norm1_mha, self.dWo, self.dWq, self.dWk, self.dWv = self.mha_backward(d_mha_out, self.cache)

        dX_embed_from_mha, self.dgamma_norm1, self.dbeta_norm1 = self.norm1.backward(dX_norm1_mha)

        dX_embed_total = dX_embed_from_mha + dX_embed_shortcut

        self.dW_embed = torch.zeros_like(self.embedding)

        tokens_flat = self.cache['X_tokens'].reshape(-1).to(device=DEVICE)
        grad_flat = dX_embed_total.reshape(-1, self.d_model)

        self.dW_embed.index_add_(0, tokens_flat, grad_flat)

    def mha_backward(self, d_mha_out, cache):
        d_attn_out, dWo = self.mhd.backward_concatenateWo(d_mha_out, cache['attention_out'], self.Wo)

        d_k = self.d_model // 32

        dQ_split, dK_split, dV_split = self.sa.backward_self_attention(d_attn_out, cache['Q'], cache['K'], cache['V'], cache['attn_weights'])

        dQ = dQ_split.transpose(1, 2).reshape(d_mha_out.shape)
        dK = dK_split.transpose(1, 2).reshape(d_mha_out.shape)
        dV = dV_split.transpose(1, 2).reshape(d_mha_out.shape)

        dX_norm1, dWq, dWk, dWv = self.mha.backward(dQ, dK, dV, cache['X_norm1'], self.Wq, self.Wk, self.Wv)

        return dX_norm1, dWo, dWq, dWk, dWv

    def get_params(self):
        return [
            self.embedding,
            self.norm1.gamma, self.norm1.beta,
            self.Wq, self.Wk, self.Wv, self.Wo,
            self.norm2.gamma, self.norm2.beta,
            self.ffn.W1, self.ffn.b1, self.ffn.W2, self.ffn.b2,
            self.norm3.gamma, self.norm3.beta,
            self.W_unembedding, self.b_unembedding
        ]

    def get_params_grads(self):
        return [
            (self.embedding, self.dW_embed),
            (self.norm1.gamma, self.dgamma_norm1),
            (self.norm1.beta, self.dbeta_norm1),
            (self.Wq, self.dWq),
            (self.Wk, self.dWk),
            (self.Wv, self.dWv),
            (self.Wo, self.dWo),
            (self.norm2.gamma, self.dgamma_norm2),
            (self.norm2.beta, self.dbeta_norm2),
            (self.ffn.W1, self.ffn.dW1),
            (self.ffn.b1, self.ffn.db1),
            (self.ffn.W2, self.ffn.dW2),
            (self.ffn.b2, self.ffn.db2),
            (self.norm3.gamma, self.dgamma_norm3),
            (self.norm3.beta, self.dbeta_norm3),
            (self.W_unembedding, self.dW_une),
            (self.b_unembedding, self.db_une)
        ]




