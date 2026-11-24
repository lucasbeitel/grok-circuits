import torch

from src.data import make_dataset


def test_shapes_and_split_sizes():
    p = 13
    ds = make_dataset(p=p, train_frac=0.3, seed=0)
    n_total = p * p
    assert ds["train_inputs"].shape[0] + ds["test_inputs"].shape[0] == n_total
    assert ds["train_inputs"].shape[1] == 3
    assert ds["vocab_size"] == p + 1


def test_targets_are_correct_mod_sum():
    p = 13
    ds = make_dataset(p=p, train_frac=0.5, seed=0)
    for inputs, targets in [
        (ds["train_inputs"], ds["train_targets"]),
        (ds["test_inputs"], ds["test_targets"]),
    ]:
        a, b, eq = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        assert torch.all(eq == p)
        assert torch.equal((a + b) % p, targets)


def test_split_is_deterministic_given_seed():
    p = 13
    ds1 = make_dataset(p=p, train_frac=0.3, seed=42)
    ds2 = make_dataset(p=p, train_frac=0.3, seed=42)
    assert torch.equal(ds1["train_inputs"], ds2["train_inputs"])
