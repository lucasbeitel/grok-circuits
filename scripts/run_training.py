"""Train the model and save checkpoint + metrics to results/."""

from pathlib import Path

from src.train import TrainConfig, train

if __name__ == "__main__":
    cfg = TrainConfig()
    history = train(cfg, Path("results"))
    print(f"final train acc: {history['train_acc'][-1]:.3f}")
    print(f"final test acc:  {history['test_acc'][-1]:.3f}")
