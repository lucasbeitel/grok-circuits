import torch

from src.model import GrokTransformer
from src.patching import forward_with_patch, sample_clean_corrupt_pairs


def make_model():
    return GrokTransformer(vocab_size=14, seq_len=3, d_model=16, n_heads=2, d_mlp=32)


def test_no_patch_matches_plain_forward():
    model = make_model()
    tokens = torch.randint(0, 14, (5, 3))
    assert torch.allclose(forward_with_patch(model, tokens), model(tokens), atol=1e-6)


def test_patching_a_runs_own_cache_is_a_no_op():
    model = make_model()
    tokens = torch.randint(0, 14, (5, 3))
    cache = {}
    plain_logits = model(tokens, cache=cache)
    for hook in ["resid_pre", "attn_out", "mlp_out", "resid_post"]:
        patched = forward_with_patch(model, tokens, patch={hook: cache[hook]})
        assert torch.allclose(patched, plain_logits, atol=1e-5)


def test_sample_pairs_have_different_answers():
    pairs = sample_clean_corrupt_pairs(p=13, n_pairs=50, seed=0)
    assert len(pairs) == 50
    for a1, b1, c1, a2, b2, c2 in pairs:
        assert c1 != c2
        assert (a1 + b1) % 13 == c1
        assert (a2 + b2) % 13 == c2
