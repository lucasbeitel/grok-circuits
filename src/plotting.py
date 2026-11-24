"""Shared matplotlib helpers, kept separate from analysis logic."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_grokking_curve(history: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["step"], history["train_acc"], label="train accuracy")
    ax.plot(history["step"], history["test_acc"], label="test accuracy")
    ax.set_xlabel("training step")
    ax.set_ylabel("accuracy")
    ax.set_title("Grokking on modular addition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
