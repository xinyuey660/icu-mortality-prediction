"""Reusable trainers with validation AUPRC early stopping."""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from .preprocessing import SequenceBatch, build_triplets


@dataclass
class PredictionResult:
    validation_probabilities: np.ndarray
    test_probabilities: np.ndarray
    best_epoch: int
    history: list[dict[str, float]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.set_num_threads(int(os.environ.get("ICU_TORCH_THREADS", "1")))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch allows setting inter-op threads only before parallel work starts.
            pass
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def get_device(requested: str = "auto"):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sequence_loader(batch: SequenceBatch, batch_size: int, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(
        torch.as_tensor(batch.values, dtype=torch.float32),
        torch.as_tensor(batch.mask, dtype=torch.float32),
        torch.as_tensor(batch.delta, dtype=torch.float32),
        torch.as_tensor(batch.static, dtype=torch.float32),
        torch.as_tensor(batch.y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _triplet_loader(batch: SequenceBatch, batch_size: int, shuffle: bool, max_tokens: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    feature_ids, values, times, padding_mask = build_triplets(batch, max_tokens=max_tokens)
    dataset = TensorDataset(
        torch.as_tensor(feature_ids, dtype=torch.long),
        torch.as_tensor(values, dtype=torch.float32),
        torch.as_tensor(times, dtype=torch.float32),
        torch.as_tensor(padding_mask, dtype=torch.bool),
        torch.as_tensor(batch.static, dtype=torch.float32),
        torch.as_tensor(batch.y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _forward(model, items: tuple[Any, ...], triplet: bool):
    if triplet:
        feature_ids, values, times, padding_mask, static, labels = items
        logits = model(feature_ids, values, times, padding_mask, static)
    else:
        values, mask, delta, static, labels = items
        logits = model(values, mask, delta, static)
    return logits, labels


def _predict(model, loader, device, triplet: bool) -> np.ndarray:
    import torch

    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for items in loader:
            items = tuple(item.to(device) for item in items)
            logits, _ = _forward(model, items, triplet)
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs)


def train_torch_model(
    model,
    train: SequenceBatch,
    validation: SequenceBatch,
    test: SequenceBatch,
    *,
    triplet: bool = False,
    max_tokens: int = 512,
    epochs: int = 80,
    patience: int = 12,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "auto",
) -> PredictionResult:
    """Train one neural branch; test labels are never used during fitting."""

    import torch

    set_seed(seed)
    active_device = get_device(device)
    model = model.to(active_device)
    loader_factory = _triplet_loader if triplet else _sequence_loader
    loader_kwargs = {"max_tokens": max_tokens} if triplet else {}
    train_loader = loader_factory(train, batch_size, True, **loader_kwargs)
    validation_loader = loader_factory(validation, batch_size, False, **loader_kwargs)
    test_loader = loader_factory(test, batch_size, False, **loader_kwargs)

    positives = float(train.y.sum())
    negatives = float(len(train.y) - positives)
    positive_weight = negatives / max(positives, 1.0)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=active_device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_state = copy.deepcopy(model.state_dict())
    best_score = -np.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for items in train_loader:
            items = tuple(item.to(active_device) for item in items)
            optimizer.zero_grad(set_to_none=True)
            logits, labels = _forward(model, items, triplet)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_probabilities = _predict(model, validation_loader, active_device, triplet)
        score = float(average_precision_score(validation.y, validation_probabilities))
        history.append({"epoch": float(epoch), "loss": float(np.mean(losses)), "val_auprc": score})
        print(
            f"  epoch={epoch:03d} loss={history[-1]['loss']:.4f} "
            f"val_auprc={score:.4f}",
            flush=True,
        )
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    validation_probabilities = _predict(model, validation_loader, active_device, triplet)
    test_probabilities = _predict(model, test_loader, active_device, triplet)
    return PredictionResult(
        validation_probabilities=validation_probabilities,
        test_probabilities=test_probabilities,
        best_epoch=best_epoch,
        history=history,
    )


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    seed: int = 42,
):
    """Regularized XGBoost tuned for a small, imbalanced clinical cohort."""

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install the training extra: pip install '.[train]'") from exc

    positives = float(np.sum(y_train))
    negatives = float(len(y_train) - positives)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=800,
        learning_rate=0.025,
        max_depth=3,
        min_child_weight=5,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=5.0,
        scale_pos_weight=negatives / max(positives, 1.0),
        early_stopping_rounds=60,
        random_state=seed,
        tree_method="hist",
        # One thread avoids libomp contention when XGBoost and PyTorch train in
        # the same macOS process. This cohort is small enough that the runtime
        # cost is minor and reproducibility improves.
        n_jobs=1,
    )
    model.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False)
    return model
