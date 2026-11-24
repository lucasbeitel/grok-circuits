"""Full-batch training loop that reproduces grokking on modular addition.

Grokking (Power et al., 2022) is the phenomenon where a network reaches
~100% train accuracy almost immediately, while test accuracy stays near
chance for a long time, then rises sharply much later in training. It
only shows up with a small train fraction and strong weight decay, which
is why those are the defaults here rather than typical training knobs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data import make_dataset
from src.model import GrokTransformer


@dataclass
class TrainConfig:
    p: int = 59
    train_frac: float = 0.6
    d_model: int = 64
    n_heads: int = 2
    d_mlp: int = 256
    lr: float = 1e-3
    weight_decay: float = 1.0
    steps: int = 3_000
    log_every: int = 50
    seed: int = 0


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits[:, -1, :].argmax(dim=-1)
    return (preds == targets).float().mean().item()


def train(cfg: TrainConfig, out_dir: Path) -> dict:
    torch.manual_seed(cfg.seed)
    ds = make_dataset(p=cfg.p, train_frac=cfg.train_frac, seed=cfg.seed)

    model = GrokTransformer(
        vocab_size=ds["vocab_size"],
        seq_len=ds["seq_len"],
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_mlp=cfg.d_mlp,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))

    history = {"step": [], "train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    for step in range(cfg.steps):
        model.train()
        logits = model(ds["train_inputs"])
        loss = F.cross_entropy(logits[:, -1, :], ds["train_targets"])
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            model.eval()
            with torch.no_grad():
                train_logits = model(ds["train_inputs"])
                test_logits = model(ds["test_inputs"])
                test_loss = F.cross_entropy(test_logits[:, -1, :], ds["test_targets"])
                history["step"].append(step)
                history["train_loss"].append(loss.item())
                history["test_loss"].append(test_loss.item())
                history["train_acc"].append(accuracy(train_logits, ds["train_targets"]))
                history["test_acc"].append(accuracy(test_logits, ds["test_targets"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "checkpoint.pt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"config": asdict(cfg), "history": history}, f)

    return history
