import torch


class SelfAttention:
    def __init__(self, d_model):
        self.d_model = d_model

    def forward(self, Q, K, V, causal_mask):
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

        final_mask = causal_mask

        scores_masked = scores + final_mask

        attn_weights = torch.softmax(scores_masked, dim=-1)

        attention_out = torch.matmul(attn_weights, V)

        return attention_out, attn_weights

    def backward_self_attention(self, d_attn_out, Q, K, V, attn_weights):
        d_k = Q.shape[-1]

        attn_weights_T = attn_weights.transpose(-2, -1).contiguous()
        dV = torch.matmul(attn_weights_T, d_attn_out)

        V_T = V.transpose(-2, -1).contiguous()
        dA = torch.matmul(d_attn_out, V_T)

        sum_dA_A = torch.sum(dA * attn_weights, dim=-1, keepdim=True)
        dS = attn_weights * (dA - sum_dA_A)
        dS = dS / (d_k ** 0.5)

        dQ = torch.matmul(dS, K)

        dS_T = dS.transpose(-2, -1).contiguous()
        dK = torch.matmul(dS_T, Q)

        return dQ, dK, dV