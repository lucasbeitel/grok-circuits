"""Find the model's key Fourier frequencies, then probe how linearly
decodable the true sum is at each point in the residual stream."""

import json

import numpy as np

from src.data import make_dataset
from src.plotting import plot_frequency_norms, plot_probe_heatmap
from src.probing import (
    collect_last_token_activations,
    embedding_frequency_norms,
    load_model,
    run_probe_sweep,
)

N_KEY_FREQS = 3
N_CONTROL_FREQS = 1

if __name__ == "__main__":
    with open("results/metrics.json") as f:
        cfg = json.load(f)["config"]
    model = load_model("results/checkpoint.pt", cfg)
    p = cfg["p"]

    norms = embedding_frequency_norms(model, p)
    order = np.argsort(-norms)
    key_freqs = sorted(int(k) for k in order[:N_KEY_FREQS])
    control_freqs = sorted(int(k) for k in order[N_KEY_FREQS : N_KEY_FREQS + N_CONTROL_FREQS])
    freqs = key_freqs + control_freqs
    print(f"key frequencies: {key_freqs}, control frequency: {control_freqs}")

    plot_frequency_norms(norms, key_freqs, "results/frequency_norms.png")

    ds = make_dataset(p=p, train_frac=1.0, seed=1)
    activations = collect_last_token_activations(model, ds["train_inputs"])
    targets = ds["train_targets"].numpy()

    probe_results = run_probe_sweep(activations, targets, p, freqs)
    plot_probe_heatmap(probe_results, "results/probe_heatmap.png")

    with open("results/probe_results.json", "w") as f:
        json.dump(
            {"key_freqs": key_freqs, "control_freqs": control_freqs, "r2_by_hook": probe_results},
            f,
            indent=2,
        )
    print(json.dumps(probe_results, indent=2))
