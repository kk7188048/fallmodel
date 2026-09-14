"""M3 - training loop with early stopping and MLflow logging."""
from __future__ import annotations

import copy
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from falldet.dataset import FallWindowDataset, make_class_weights
from falldet.model import FallDetectionModel


def _run_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.enable_grad() if is_train else torch.no_grad():
        for windows, labels in loader:
            windows, labels = windows.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()

            logits = model(windows)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * windows.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += windows.size(0)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def train(config: dict) -> dict:
    """Train one config end to end; returns the run summary and writes
    the best-val-loss checkpoint to `paths.checkpoints_dir`.
    """
    paths = config["paths"]
    train_cfg = config["train"]

    torch.manual_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = FallWindowDataset(paths["processed_dir"], "train")
    val_ds = FallWindowDataset(paths["processed_dir"], "val")
    if len(train_ds) == 0:
        raise ValueError(
            f"no training windows found under {paths['processed_dir']} - "
            "run scripts/run_extraction.py first"
        )

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False)

    model = FallDetectionModel.from_config(config, input_dim=train_ds.input_dim).to(device)
    class_weights = make_class_weights(train_ds).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    mlflow.set_tracking_uri(paths["mlflow_uri"])
    with mlflow.start_run():
        mlflow.log_params(
            {
                "input_dim": train_ds.input_dim,
                "train_windows": len(train_ds),
                "val_windows": len(val_ds),
                **{f"model.{k}": v for k, v in config["model"].items()},
                **{f"train.{k}": v for k, v in train_cfg.items()},
            }
        )

        for epoch in range(1, train_cfg["max_epochs"] + 1):
            train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = _run_epoch(model, val_loader, criterion, None, device)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )
            print(
                f"epoch {epoch:3d} | train_loss {train_loss:.4f} acc {train_acc:.3f} "
                f"| val_loss {val_loss:.4f} acc {val_acc:.3f}"
            )

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                    print(f"early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        checkpoint_dir = Path(paths["checkpoints_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "best_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": train_ds.input_dim,
                "config": config,
            },
            checkpoint_path,
        )
        mlflow.log_artifact(str(checkpoint_path))

    return {
        "best_val_loss": best_val_loss,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
    }
