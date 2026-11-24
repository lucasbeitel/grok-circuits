"""Shared matplotlib helpers, kept separate from analysis logic."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


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


def plot_frequency_norms(norms: np.ndarray, highlight_freqs: list[int], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:orange" if k in highlight_freqs else "tab:blue" for k in range(len(norms))]
    ax.bar(range(len(norms)), norms, color=colors)
    ax.set_xlabel("frequency k")
    ax.set_ylabel("embedding norm at frequency k")
    ax.set_title("Token embedding is concentrated in a few Fourier frequencies")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_probe_heatmap(results: dict, out_path: str) -> None:
    hooks = list(results.keys())
    freqs = list(next(iter(results.values())).keys())
    grid = np.array([[results[hook][freq] for freq in freqs] for hook in hooks])

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(freqs)), [f"k={f}" for f in freqs])
    ax.set_yticks(range(len(hooks)), hooks)
    ax.set_title("Linear probe R²: is cos/sin(sum) decodable here?")
    for i in range(len(hooks)):
        for j in range(len(freqs)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="held-out R²")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
