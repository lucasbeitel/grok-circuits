"""A minimal one-layer transformer for the modular addition task.

No LayerNorm: the original grokking-circuits literature drops it so the
residual stream stays a clean linear sum of embedding + attention + MLP
outputs, which is what makes the Fourier analysis in src/probing.py and
the activation patching in src/patching.py interpretable.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            tensor.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
            for tensor in (q, k, v)
        )
        attn = torch.softmax(q @ k.transpose(-2, -1) / self.d_head**0.5, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, d)
        return self.out(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d_model, bias=False)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class GrokTransformer(nn.Module):
    """One attention block + one MLP block, with an activation cache.

    Call `forward(x, cache={})` and the dict will be filled with the
    residual-stream tensors at each hook point, keyed by name. This is
    how src/probing.py and src/patching.py get at intermediate activations
    without needing a full hooking framework.
    """

    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.attn = Attention(d_model, n_heads)
        self.mlp = MLP(d_model, d_mlp)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        b, t = tokens.shape
        positions = torch.arange(t, device=tokens.device)
        resid = self.tok_emb(tokens) + self.pos_emb(positions)
        if cache is not None:
            cache["resid_pre"] = resid

        attn_out = self.attn(resid)
        resid = resid + attn_out
        if cache is not None:
            cache["attn_out"] = attn_out
            cache["resid_mid"] = resid

        mlp_out = self.mlp(resid)
        resid = resid + mlp_out
        if cache is not None:
            cache["mlp_out"] = mlp_out
            cache["resid_post"] = resid

        logits = self.unembed(resid)
        return logits
