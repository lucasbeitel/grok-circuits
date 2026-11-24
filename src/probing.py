"""Linear probes for the Fourier/trig-identity circuit found in grokked
modular addition networks (Nanda et al., 2023).

The model has no reason to represent numbers in order along a line; what
it actually tends to learn is a small set of frequencies k for which
cos(2*pi*k*n/p) and sin(2*pi*k*n/p) are linearly readable out of its
embeddings, because trig angle-addition identities let attention+MLP
compute cos/sin of (a+b) from cos/sin of a and b using only multiplies
and adds. We first find which frequencies the token embedding actually
uses, then probe how linearly decodable "cos/sin of the true sum" is at
each point in the residual stream.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from src.model import GrokTransformer


def load_model(checkpoint_path, config: dict) -> GrokTransformer:
    model = GrokTransformer(
        vocab_size=config["p"] + 1,
        seq_len=3,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_mlp=config["d_mlp"],
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def embedding_frequency_norms(model: GrokTransformer, p: int) -> np.ndarray:
    """Project the number-token embeddings onto each Fourier frequency.

    Returns an array of length p // 2 + 1: entry k is the norm of the
    embedding matrix's component at frequency k. A grokked model
    concentrates this energy in a handful of frequencies instead of
    spreading it evenly.
    """
    emb = model.tok_emb.weight[:p].detach().numpy()  # (p, d_model)
    n = np.arange(p)
    norms = []
    for k in range(p // 2 + 1):
        cos_basis = np.cos(2 * np.pi * k * n / p)
        sin_basis = np.sin(2 * np.pi * k * n / p)
        cos_comp = cos_basis @ emb / np.linalg.norm(cos_basis)
        sin_comp = sin_basis @ emb / np.linalg.norm(sin_basis) if k not in (0, p / 2) else np.zeros_like(cos_comp)
        norms.append(np.linalg.norm(cos_comp) + np.linalg.norm(sin_comp))
    return np.array(norms)


def collect_last_token_activations(model: GrokTransformer, inputs: torch.Tensor) -> dict:
    """Run the model and return each hook's residual stream at the final
    ("=") token position, which is where the answer is read out."""
    cache = {}
    with torch.no_grad():
        model(inputs, cache=cache)
    return {name: tensor[:, -1, :].numpy() for name, tensor in cache.items()}


def probe_r2_for_frequency(activations: np.ndarray, targets: np.ndarray, p: int, k: int, seed: int = 0) -> float:
    """Ridge-regress [cos(2*pi*k*target/p), sin(2*pi*k*target/p)] from
    activations and return held-out R^2, averaged over the two components."""
    labels = np.stack(
        [np.cos(2 * np.pi * k * targets / p), np.sin(2 * np.pi * k * targets / p)], axis=1
    )
    x_train, x_test, y_train, y_test = train_test_split(
        activations, labels, test_size=0.3, random_state=seed
    )
    reg = Ridge(alpha=1.0).fit(x_train, y_train)
    return reg.score(x_test, y_test)


def run_probe_sweep(activations_by_hook: dict, targets: np.ndarray, p: int, freqs: list[int]) -> dict:
    """Return {hook_name: {freq: r2}} for every hook point and frequency."""
    results = {}
    for hook_name, activations in activations_by_hook.items():
        results[hook_name] = {
            k: probe_r2_for_frequency(activations, targets, p, k) for k in freqs
        }
    return results
