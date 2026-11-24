from pathlib import Path

from src.train import TrainConfig, train


def test_training_runs_and_saves_outputs(tmp_path):
    cfg = TrainConfig(p=13, train_frac=0.6, d_model=16, n_heads=2, d_mlp=32, steps=20, log_every=5)
    history = train(cfg, tmp_path)
    assert (tmp_path / "checkpoint.pt").exists()
    assert (tmp_path / "metrics.json").exists()
    assert len(history["step"]) > 0
    assert 0.0 <= history["train_acc"][-1] <= 1.0
