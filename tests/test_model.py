import torch

from src.model import GrokTransformer


def make_model(vocab_size=14, seq_len=3):
    return GrokTransformer(vocab_size=vocab_size, seq_len=seq_len, d_model=32, n_heads=4, d_mlp=64)


def test_forward_output_shape():
    model = make_model()
    tokens = torch.randint(0, 14, (8, 3))
    logits = model(tokens)
    assert logits.shape == (8, 3, 14)


def test_cache_is_populated_with_expected_keys():
    model = make_model()
    tokens = torch.randint(0, 14, (8, 3))
    cache = {}
    model(tokens, cache=cache)
    assert set(cache.keys()) == {"resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"}
    for tensor in cache.values():
        assert tensor.shape == (8, 3, 32)


def test_residual_stream_is_additive():
    model = make_model()
    tokens = torch.randint(0, 14, (4, 3))
    cache = {}
    model(tokens, cache=cache)
    assert torch.allclose(cache["resid_mid"], cache["resid_pre"] + cache["attn_out"], atol=1e-5)
    assert torch.allclose(cache["resid_post"], cache["resid_mid"] + cache["mlp_out"], atol=1e-5)
