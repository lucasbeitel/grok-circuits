import numpy as np

from src.probing import embedding_frequency_norms, probe_r2_for_frequency
from src.model import GrokTransformer


def test_frequency_norms_peak_at_the_true_frequency():
    p, k0 = 23, 5
    model = GrokTransformer(vocab_size=p + 1, seq_len=3, d_model=4, n_heads=2, d_mlp=8)
    n = np.arange(p)
    pure_tone = np.stack([np.cos(2 * np.pi * k0 * n / p), np.sin(2 * np.pi * k0 * n / p)], axis=1)
    model.tok_emb.weight.data[:p, :2] = torch_from(pure_tone)

    norms = embedding_frequency_norms(model, p)
    assert int(np.argmax(norms)) == k0


def torch_from(array):
    import torch

    return torch.tensor(array, dtype=torch.float32)


def test_probe_recovers_linearly_encoded_signal():
    rng = np.random.default_rng(0)
    p, k, n = 23, 3, 500
    targets = rng.integers(0, p, size=n)
    signal = np.stack([np.cos(2 * np.pi * k * targets / p), np.sin(2 * np.pi * k * targets / p)], axis=1)
    activations = np.concatenate([signal, rng.normal(scale=0.01, size=(n, 10))], axis=1)

    r2 = probe_r2_for_frequency(activations, targets, p, k)
    assert r2 > 0.9


def test_probe_is_low_for_unrelated_noise():
    rng = np.random.default_rng(0)
    p, k, n = 23, 3, 500
    targets = rng.integers(0, p, size=n)
    activations = rng.normal(size=(n, 10))

    r2 = probe_r2_for_frequency(activations, targets, p, k)
    assert r2 < 0.3
