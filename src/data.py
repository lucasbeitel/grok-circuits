"""Dataset generation for the modular addition task.

Each example is the sequence [a, b, EQUALS] -> target c = (a + b) mod p.
Using a prime modulus p keeps Z/pZ a field, which is what makes the
Fourier analysis in src/probing.py well-behaved.
"""

from __future__ import annotations

import torch


def make_dataset(p: int, train_frac: float, seed: int = 0):
    """Build the full (a, b) -> (a+b) mod p dataset and split it.

    Returns a dict with input/target tensors for the train and test splits,
    plus the vocab size (p numbers + 1 "=" token).
    """
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    equals = torch.full_like(a, fill_value=p)  # the "=" token, id == p

    inputs = torch.stack([a, b, equals], dim=1)  # (p*p, 3)
    targets = (a + b) % p

    n = inputs.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = int(n * train_frac)

    train_idx, test_idx = perm[:n_train], perm[n_train:]
    return {
        "vocab_size": p + 1,
        "seq_len": 3,
        "train_inputs": inputs[train_idx],
        "train_targets": targets[train_idx],
        "test_inputs": inputs[test_idx],
        "test_targets": targets[test_idx],
    }
