import torch

class MultiHeadAttention:
    @staticmethod
    def forward( final_embedding, Wq, Wk, Wv):
        Q = torch.matmul(final_embedding, Wq)
        K = torch.matmul(final_embedding, Wk)
        V = torch.matmul(final_embedding, Wv)

        return Q, K, V

    @staticmethod
    def backward(dQ, dK, dV, X_norm1, Wq, Wk, Wv):
        shape_original = X_norm1.shape
        X_flat = X_norm1.reshape(-1, X_norm1.shape[-1])
        dQ_flat = dQ.reshape(-1, dQ.shape[-1])
        dK_flat = dK.reshape(-1, dK.shape[-1])
        dV_flat = dV.reshape(-1, dV.shape[-1])

        dWq = torch.matmul(X_flat.T, dQ_flat)
        dWk = torch.matmul(X_flat.T, dK_flat)
        dWv = torch.matmul(X_flat.T, dV_flat)

        dX_q = torch.matmul(dQ_flat, Wq.T)
        dX_k = torch.matmul(dK_flat, Wk.T)
        dX_v = torch.matmul(dV_flat, Wv.T)

        dX_flat = dX_q + dX_k + dX_v
        dX_norm1 = dX_flat.reshape(shape_original)

        return dX_norm1, dWq, dWk, dWv


class MultiHeadDivisor:
    def __init__(self, d_model):
        self.d_model = d_model

    def rearrange(self, current_batch_size, Q, K, V, num_heads=32):
        batch_size, seq_len, d_model = Q.shape
        d_k = d_model // num_heads

        Q_split = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        K_split = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        V_split = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(1, 2)

        return Q_split, K_split, V_split

    def concatenateWo(self, current_batch_size, attention_out, Wo):
        batch_size, num_heads, seq_len, d_k = attention_out.shape

        attention_concatenated = attention_out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        multi_head_output = torch.matmul(attention_concatenated, Wo)

        return multi_head_output

    def backward_concatenateWo(self, d_mha_out, attention_out, Wo):
        batch_size, num_heads, seq_len, d_k = attention_out.shape

        attn_concatenated_flat = attention_out.transpose(1, 2).reshape(-1, self.d_model)
        d_mha_out_flat = d_mha_out.reshape(-1, self.d_model)

        dWo = torch.matmul(attn_concatenated_flat.T, d_mha_out_flat)

        d_attn_concatenated = torch.matmul(d_mha_out_flat, Wo.T)

        d_attn_out = d_attn_concatenated.reshape(batch_size, seq_len, num_heads, d_k).transpose(1, 2)

        return d_attn_out, dWo

