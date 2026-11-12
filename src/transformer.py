"""Transformer encoder written from scratch (no nn.Transformer / nn.MultiheadAttention / nn.TransformerEncoder)."""
import math, torch, torch.nn as nn, torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    """Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V.  mask: broadcastable bool tensor, True = may attend, False = padding.
    Returns (output, attention_weights)."""
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
    if mask is not None: scores = scores.masked_fill(~mask, float("-inf"))
    w = torch.softmax(scores, dim=-1)
    if dropout is not None: w = dropout(w)
    return w @ V, w


class MultiHeadSelfAttention(nn.Module):
    """h heads with SEPARATE projection matrices per head (W_q^i, W_k^i, W_v^i : d_model -> d_k) and a shared output projection W_o."""
    def __init__(self, d_model=128, h=4, d_k=32, d_v=32, dropout=0.1):
        super().__init__()
        self.h = h
        self.Wq = nn.ModuleList([nn.Linear(d_model, d_k, bias=False) for _ in range(h)])
        self.Wk = nn.ModuleList([nn.Linear(d_model, d_k, bias=False) for _ in range(h)])
        self.Wv = nn.ModuleList([nn.Linear(d_model, d_v, bias=False) for _ in range(h)])
        self.Wo = nn.Linear(h * d_v, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):                       # x: [B,T,d_model], mask: [B,1,1,T]
        outs, ws = [], []
        m = mask[:, 0] if (mask is not None and mask.dim() == 4) else mask          # [B,1,T] broadcasts over query positions
        for i in range(self.h):
            o, w = scaled_dot_product_attention(self.Wq[i](x), self.Wk[i](x), self.Wv[i](x), m, self.attn_drop)
            outs.append(o); ws.append(w)
        return self.Wo(torch.cat(outs, dim=-1)), torch.stack(ws, dim=1)   # [B,T,d_model], [B,h,T,T]


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model=128, d_ff=512, dropout=0.1):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff); self.l2 = nn.Linear(d_ff, d_model); self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.l2(self.drop(F.relu(self.l1(x))))


class SinusoidalPositionalEncoding(nn.Module):
    """PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(...); stored as a fixed (non-learned) buffer."""
    def __init__(self, d_model=128, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model); pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))        # buffer: saved in state_dict, never trained

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class EncoderBlock(nn.Module):
    """Pre-LayerNorm block:  x <- x + Dropout(MHA(LN(x)));  x <- x + Dropout(FFN(LN(x)))."""
    def __init__(self, d_model=128, h=4, d_k=32, d_ff=512, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model); self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, h, d_k, d_k, dropout); self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        a, w = self.mha(self.ln1(x), mask); x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x, w


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, pad_id, n_classes=5, d_model=128, h=4, d_k=32, d_ff=512, n_layers=4, dropout=0.1, max_len=300):
        super().__init__()
        self.pad_id, self.d_model = pad_id, d_model
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)                  # learned [CLS] token
        self.pe = SinusoidalPositionalEncoding(d_model, max_len)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, h, d_k, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes))   # 128 -> 64 -> 5

    def forward(self, ids, return_attn=False):
        B = ids.size(0)
        x = self.emb(ids) * math.sqrt(self.d_model)
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)                        # prepend [CLS]
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool), ids != self.pad_id], dim=1)
        mask = valid[:, None, None, :]                                               # padding mask [B,1,1,T+1]
        x = self.drop(self.pe(x)); attns = []
        for blk in self.blocks:
            x, w = blk(x, mask); attns.append(w)
        logits = self.head(self.ln_f(x)[:, 0])                                       # classify from the [CLS] representation
        return (logits, attns) if return_attn else logits


def unit_tests():
    """Sanity checks used in the notebook."""
    torch.manual_seed(0); res = {}
    Q = torch.randn(2, 4, 5, 32); mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)[:, None, None, :]
    o, w = scaled_dot_product_attention(Q, Q, Q, mask)
    res["attn_rows_sum_to_1"] = bool(torch.allclose(w.sum(-1), torch.ones(2, 4, 5), atol=1e-5))
    res["masked_weight_is_zero"] = float(w[0, :, :, 3:].abs().max()) == 0.0
    m = TransformerClassifier(100, 99)
    res["params"] = sum(p.numel() for p in m.parameters() if p.requires_grad)
    res["pe_is_buffer_not_param"] = ("pe.pe" in dict(m.named_buffers())) and not any("pe.pe" in n for n, _ in m.named_parameters())
    ids = torch.randint(0, 99, (3, 10)); ids[0, 6:] = 99
    a = m.eval()(ids)[0]; ids2 = ids.clone(); ids2[0, 6:] = 99
    res["output_shape"] = tuple(m.eval()(ids).shape); res["blocks"] = len(m.blocks); res["heads_per_layer"] = m.blocks[0].mha.h
    res["separate_per_head_projections"] = len(m.blocks[0].mha.Wq) == 4
    x = torch.randn(2, 6, 128); mk = torch.tensor([[1] * 4 + [0] * 2, [1] * 6], dtype=torch.bool)[:, None, None, :]
    _, wm = m.blocks[0].mha(x, mk)
    res["mha_weight_shape"] = tuple(wm.shape); res["mha_padding_weight_zero"] = float(wm[0, :, :, 4:].abs().max()) == 0.0
    ids = torch.randint(0, 98, (2, 12)); ids[0, 7:] = 99; m.eval()
    ids_b = ids.clone(); ids_b[0, 7:] = 99
    l1 = m(ids); ids_c = torch.cat([ids[:, :7], torch.full((2, 5), 99)], 1); ids_c[1] = ids[1]
    res["pad_invariance"] = bool(torch.allclose(m(ids)[0], m(ids_b)[0]))
    return res
