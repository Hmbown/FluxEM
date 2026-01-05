"""Training loop for baseline models."""

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .tokenizer import CharTokenizer


class ArithmeticDataset(Dataset):
    """Dataset for arithmetic expressions."""

    def __init__(self, samples: List[dict], tokenizer: CharTokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids, mask = self.tokenizer.encode_batch([sample["expression"]])
        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": mask.squeeze(0),
            "target": torch.tensor(sample["ground_truth"], dtype=torch.float32),
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
    }


class Trainer:
    """Trainer for arithmetic regression models."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: CharTokenizer,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        use_log_scale: bool = True,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.use_log_scale = use_log_scale

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.loss_fn = nn.MSELoss()

    def _scale_target(self, y: torch.Tensor) -> torch.Tensor:
        """Apply log scaling: sign(y) * log(1 + |y|)."""
        if not self.use_log_scale:
            return y
        return torch.sign(y) * torch.log1p(torch.abs(y))

    def _unscale_prediction(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse: sign(y) * (exp(|y|) - 1)."""
        if not self.use_log_scale:
            return y
        return torch.sign(y) * (torch.exp(torch.abs(y)) - 1)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target = batch["target"].to(self.device)

            scaled_target = self._scale_target(target)

            self.optimizer.zero_grad()
            pred = self.model(input_ids, attention_mask)
            loss = self.loss_fn(pred, scaled_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()

        all_preds = []
        all_targets = []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target = batch["target"]

            pred = self.model(input_ids, attention_mask)
            pred = self._unscale_prediction(pred).cpu()

            all_preds.append(pred)
            all_targets.append(target)

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        # Metrics
        mae = torch.abs(preds - targets).mean().item()
        rel_errors = torch.abs(preds - targets) / (torch.abs(targets) + 1e-8)
        mean_rel_error = rel_errors.mean().item()
        within_tol = (rel_errors < 0.01).float().mean().item()

        return {
            "mae": mae,
            "mean_relative_error": mean_rel_error,
            "accuracy_1pct": within_tol,
        }

    def train(
        self,
        train_samples: List[dict],
        val_samples: List[dict],
        epochs: int = 50,
        batch_size: int = 64,
        checkpoint_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Full training loop."""
        train_dataset = ArithmeticDataset(train_samples, self.tokenizer)
        val_dataset = ArithmeticDataset(val_samples, self.tokenizer)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        history = {"train_loss": [], "val_mae": [], "val_acc": []}
        best_val_mae = float("inf")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            self.scheduler.step(val_metrics["mae"])

            history["train_loss"].append(train_loss)
            history["val_mae"].append(val_metrics["mae"])
            history["val_acc"].append(val_metrics["accuracy_1pct"])

            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"loss={train_loss:.4f}, "
                    f"val_mae={val_metrics['mae']:.2f}, "
                    f"val_acc={val_metrics['accuracy_1pct']:.1%}"
                )

            # Save lowest validation MAE model
            if checkpoint_dir and val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), checkpoint_dir / "best_model.pt")

        return history
