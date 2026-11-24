"""Activation patching: a causal complement to the linear probes.

Probing (src/probing.py) only shows correlation — a hook point's activations
happen to be linearly predictive of the sum. Patching shows causation: we
take a "clean" input the model answers correctly, a "corrupted" input with a
different answer, and splice the clean run's activation at one hook into an
otherwise-corrupted forward pass. If that single substitution flips the
model back to the clean answer, that hook is where the causally relevant
computation lives (Zhang & Nanda, 2023 style denoising patch).
"""

from __future__ import annotations

import numpy as np
import torch

from src.model import GrokTransformer

HOOK_NAMES = ["resid_pre", "attn_out", "mlp_out", "resid_post"]


def _splice_last_position(current: torch.Tensor, override: torch.Tensor) -> torch.Tensor:
    """Replace only the final ("=") token's slice with the override, leaving
    the a/b token positions as whatever the current (corrupted) run computed."""
    out = current.clone()
    out[:, -1, :] = override[:, -1, :]
    return out


def forward_with_patch(model: GrokTransformer, tokens: torch.Tensor, patch: dict | None = None) -> torch.Tensor:
    """Replay the model's forward pass, substituting one hook's activation
    at the "=" token position only. The a/b positions always use whatever
    `tokens` naturally produces, so a patch isolates the causal effect of
    that single hook rather than silently re-running the clean input."""
    patch = patch or {}
    b, t = tokens.shape
    positions = torch.arange(t, device=tokens.device)
    resid = model.tok_emb(tokens) + model.pos_emb(positions)
    if "resid_pre" in patch:
        resid = _splice_last_position(resid, patch["resid_pre"])

    attn_out = model.attn(resid)
    if "attn_out" in patch:
        attn_out = _splice_last_position(attn_out, patch["attn_out"])
    resid = resid + attn_out
    if "resid_mid" in patch:
        resid = _splice_last_position(resid, patch["resid_mid"])

    mlp_out = model.mlp(resid)
    if "mlp_out" in patch:
        mlp_out = _splice_last_position(mlp_out, patch["mlp_out"])
    resid = resid + mlp_out
    if "resid_post" in patch:
        resid = _splice_last_position(resid, patch["resid_post"])

    return model.unembed(resid)


def sample_clean_corrupt_pairs(p: int, n_pairs: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    pairs = []
    while len(pairs) < n_pairs:
        a1, b1, a2, b2 = rng.integers(0, p, size=4)
        c1, c2 = (a1 + b1) % p, (a2 + b2) % p
        if c1 != c2:
            pairs.append((int(a1), int(b1), int(c1), int(a2), int(b2), int(c2)))
    return pairs


def _logit_diff(logits: torch.Tensor, clean_ans: torch.Tensor, corrupt_ans: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(logits.shape[0])
    last = logits[:, -1, :]
    return last[idx, clean_ans] - last[idx, corrupt_ans]


def run_patching_experiment(model: GrokTransformer, p: int, n_pairs: int = 200, seed: int = 0) -> dict:
    """For each hook, patch it with the clean run's value and report how
    much of the clean-vs-corrupt logit gap gets recovered, averaged over
    many random clean/corrupt pairs. 0 = no effect, 1 = full recovery."""
    pairs = sample_clean_corrupt_pairs(p, n_pairs, seed)
    clean_tokens = torch.tensor([[a1, b1, p] for a1, b1, c1, a2, b2, c2 in pairs])
    corrupt_tokens = torch.tensor([[a2, b2, p] for a1, b1, c1, a2, b2, c2 in pairs])
    clean_ans = torch.tensor([c1 for a1, b1, c1, a2, b2, c2 in pairs])
    corrupt_ans = torch.tensor([c2 for a1, b1, c1, a2, b2, c2 in pairs])

    with torch.no_grad():
        clean_cache: dict = {}
        clean_logits = model(clean_tokens, cache=clean_cache)
        corrupt_logits = model(corrupt_tokens)

        clean_ld = _logit_diff(clean_logits, clean_ans, corrupt_ans).mean().item()
        corrupt_ld = _logit_diff(corrupt_logits, clean_ans, corrupt_ans).mean().item()

        recovery_by_hook = {}
        for hook in HOOK_NAMES:
            patched_logits = forward_with_patch(model, corrupt_tokens, patch={hook: clean_cache[hook]})
            patched_ld = _logit_diff(patched_logits, clean_ans, corrupt_ans).mean().item()
            recovery_by_hook[hook] = (patched_ld - corrupt_ld) / (clean_ld - corrupt_ld)

    return {
        "clean_logit_diff": clean_ld,
        "corrupt_logit_diff": corrupt_ld,
        "recovery_by_hook": recovery_by_hook,
    }
