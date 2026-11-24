"""Run the activation patching experiment against the trained checkpoint."""

import json

from src.patching import run_patching_experiment
from src.plotting import plot_patching_recovery
from src.probing import load_model

if __name__ == "__main__":
    with open("results/metrics.json") as f:
        cfg = json.load(f)["config"]
    model = load_model("results/checkpoint.pt", cfg)

    results = run_patching_experiment(model, p=cfg["p"], n_pairs=500, seed=0)
    print(json.dumps(results, indent=2))

    plot_patching_recovery(results["recovery_by_hook"], "results/patching_recovery.png")
    with open("results/patching_results.json", "w") as f:
        json.dump(results, f, indent=2)
