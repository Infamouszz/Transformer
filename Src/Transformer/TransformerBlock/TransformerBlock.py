from Src.Transformer.MultiHeadAttention import AttentionWeightsInitializer
from Src.Transformer.MultiHeadAttention.MultiHead import MultiHeadAttention
from Src.Transformer.MultiHeadAttention.MultiHead import MultiHeadDivisor
from Src.Transformer.Mask.CausalMask import CausalMask
from Src.Transformer.MultiHeadAttention.SelfAttention import SelfAttention
from Src.Transformer.ResidualConnection import ResidualConnectionCalculator
from Src.Transformer.NeuralNetwork import NeuralNetwork
from Src.Transformer.Normalization import LayerNormalization

class TransformerBlock:
    def __init__(self, d_model, causal_mask_size, max_seq_len):
        self.seq_len = max_seq_len

        self.res = ResidualConnectionCalculator.ResidualConnectionCalculator
        self.norm1 = LayerNormalization.LayerNorm(d_model)
        self.norm2 = LayerNormalization.LayerNorm(d_model)
        self.mha = MultiHeadAttention
        self.mhd = MultiHeadDivisor(d_model)
        self.mask = CausalMask.initialize_causal_mask(causal_mask_size)
        self.sa = SelfAttention(d_model)
        self.ffn = NeuralNetwork.NeuralNetwork(d_model, d_model * 4)

        self.Wq, self.Wk, self.Wv, self.Wo = AttentionWeightsInitializer.WeightInitializer(d_model).init_weights_HE()

        self.cache = {}


    def forward(self, input_X):
        batch_size, seq_len, _ = input_X.shape

        X_norm1 = self.norm1.forward(input_X)
        Q, K, V = self.mha.forward(X_norm1, self.Wq, self.Wk, self.Wv)

        Q_split, K_split, V_split = self.mhd.rearrange(Q, K, V, num_heads=32)

        causal_mask_sliced = self.mask[:, :, :seq_len, :seq_len]

        attention_out, attn_weights = self.sa.forward(Q_split, K_split, V_split, causal_mask_sliced)
        mha_out = self.mhd.concatenateWo(attention_out, self.Wo)

        res1 = self.res.calculate(input_X, mha_out)
        X_norm2 = self.norm2.forward(res1)

        y_pred_ffn, ffn_hidden_act = self.ffn.forward(X_norm2)

        res2 = self.res.calculate(res1, y_pred_ffn)

        self.cache = {
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
        }

        return res2

    def backward(self, dZ):
        dy_pred_ffn = dZ.clone()
        dres1_shortcut = dZ.clone()

        dX_norm2 = self.ffn.backward(dy_pred_ffn)

        dres1_ffn, self.dgamma_norm2, self.dbeta_norm2 = self.norm2.backward(dX_norm2)

        dres1_total = dres1_ffn + dres1_shortcut

        dmha_out = dres1_total.clone()
        dinput_X_shortcut = dres1_total.clone()

        d_attn_out, self.dWo = self.mhd.backward_concatenateWo(dmha_out, self.cache['attention_out'], self.Wo)

        dQ_split, dK_split, dV_split = self.sa.backward_self_attention(d_attn_out, self.cache['Q'], self.cache['K'], self.cache['V'], self.cache['attn_weights'])

        dQ = dQ_split.transpose(1, 2).reshape(dZ.shape)
        dK = dK_split.transpose(1, 2).reshape(dZ.shape)
        dV = dV_split.transpose(1, 2).reshape(dZ.shape)

        dX_norm1, self.dWq, self.dWk, self.dWv = self.mha.backward(dQ, dK, dV, self.cache['X_norm1'], self.Wq, self.Wk, self.Wv)

        dinput_X_mha, self.dgamma_norm1, self.dbeta_norm1 = self.norm1.backward(dX_norm1)

        dinput_X_total = dinput_X_mha + dinput_X_shortcut

        return dinput_X_total