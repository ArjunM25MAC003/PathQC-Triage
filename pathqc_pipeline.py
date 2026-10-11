"""Reusable PathQC training and inference components.

The notebook is useful for exploration, but production-like experiments need
reusable functions with explicit configuration and deterministic behavior.
This module keeps the model, training, tile inference, and triage logic in one
testable place.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_NAMES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
ARTIFACT_CLASS_IDS = (1, 2)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class QCConfig:
    """Configuration shared by training and inference."""

    image_size: int = 64
    batch_size: int = 128
    num_workers: int = 0
    epochs: int = 20
    learning_rate: float = 3e-4
    backbone_learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    patience: int = 5
    label_smoothing: float = 0.05
    seed: int = 42
    unfreeze_last_blocks: int = 3
    artifact_review_threshold: float = 0.10
    artifact_reject_threshold: float = 0.30
    min_confidence_for_decision: float = 0.55


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_transforms(image_size: int = 64) -> tuple[transforms.Compose, transforms.Compose]:
    """Return stain-safe training and deterministic validation transforms."""

    train = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((0, 90)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    validation = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train, validation


class IndexedDataset(Dataset[tuple[Tensor, int]]):
    """Apply a transform to a subset of a MedMNIST-compatible dataset."""

    def __init__(
        self,
        dataset: Dataset[Any],
        indices: Sequence[int],
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, label = self.dataset[self.indices[index]]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        target = int(np.asarray(label).reshape(-1)[0])
        return self.transform(image.convert("RGB")), target


def balanced_indices(
    labels: Sequence[int] | np.ndarray,
    samples_per_class: int | None = None,
    seed: int = 42,
) -> list[int]:
    """Create a stratified subset without assuming all nine classes exist."""

    label_array = np.asarray(labels).reshape(-1).astype(int)
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(label_array):
        by_class.setdefault(int(label), []).append(index)

    selected: list[int] = []
    for class_indices in by_class.values():
        count = len(class_indices) if samples_per_class is None else min(
            samples_per_class, len(class_indices)
        )
        selected.extend(rng.sample(class_indices, count))
    rng.shuffle(selected)
    return selected


def build_classifier(
    num_classes: int = len(CLASS_NAMES),
    unfreeze_last_blocks: int = 3,
    pretrained: bool = True,
) -> nn.Module:
    """Build MobileNetV3 with a partially trainable backbone.

    Freezing every feature layer, as the original notebook did, limits domain
    adaptation. The final feature blocks are unfrozen so stain and morphology
    can be learned while retaining most of ImageNet's useful low-level
    features.
    """

    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    blocks = list(model.features.children())
    for block in blocks[-max(0, unfreeze_last_blocks) :]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(0.25),
        nn.Linear(256, num_classes),
    )
    return model


def _move_batch(images: Tensor, targets: Tensor, device: torch.device) -> tuple[Tensor, Tensor]:
    return images.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def _run_epoch(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    predictions: list[np.ndarray] = []
    targets_seen: list[np.ndarray] = []

    for images, targets in loader:
        images, targets = _move_batch(images, targets, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
            loss = criterion(logits, targets)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.detach().item() * targets.size(0)
        predictions.append(logits.detach().argmax(dim=1).cpu().numpy())
        targets_seen.append(targets.cpu().numpy())

    if not targets_seen:
        raise ValueError("Cannot run an epoch on an empty data loader")
    y_true = np.concatenate(targets_seen)
    y_pred = np.concatenate(predictions)
    return total_loss / len(y_true), float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    validation_loader: DataLoader[tuple[Tensor, Tensor]],
    config: QCConfig,
    device: torch.device,
    checkpoint_path: str | Path,
    class_weights: Tensor | None = None,
) -> dict[str, list[float]]:
    """Train with AMP, macro-F1 selection, early stopping, and full checkpoints."""

    seed_everything(config.seed)
    model.to(device)
    head_parameters = list(model.classifier.parameters())
    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("classifier.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": config.learning_rate},
            {"params": backbone_parameters, "lr": config.backbone_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=config.label_smoothing,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_macro_f1": [],
    }
    best_f1 = -1.0
    stale_epochs = 0

    for epoch in range(config.epochs):
        started = time.perf_counter()
        train_loss, train_f1 = _run_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        with torch.no_grad():
            val_loss, val_f1 = _run_epoch(
                model, validation_loader, criterion, None, device, scaler
            )
        scheduler.step(val_f1)
        for key, value in (
            ("train_loss", train_loss),
            ("train_macro_f1", train_f1),
            ("val_loss", val_loss),
            ("val_macro_f1", val_f1),
        ):
            history[key].append(value)
        print(
            f"Epoch {epoch + 1:02d}/{config.epochs}: "
            f"train_f1={train_f1:.4f} val_f1={val_f1:.4f} "
            f"({time.perf_counter() - started:.1f}s)"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            stale_epochs = 0
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_macro_f1": val_f1,
                "config": asdict(config),
                "class_names": CLASS_NAMES,
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

    metadata_path = Path(checkpoint_path).with_suffix(".json")
    metadata_path.write_text(json.dumps({"best_val_macro_f1": best_f1, **history}, indent=2))
    return history


def _image_to_tensor(image: Image.Image, transform: Callable[[Image.Image], Tensor]) -> Tensor:
    return transform(image.convert("RGB"))


def predict_tiles(
    image: Image.Image,
    model: nn.Module,
    transform: Callable[[Image.Image], Tensor],
    tile_size: int,
    device: torch.device,
    stride: int | None = None,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Run overlapping, batched tile inference and return probabilities/boxes."""

    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    stride = stride or tile_size
    if stride <= 0 or stride > tile_size:
        raise ValueError("stride must be between 1 and tile_size")
    width, height = image.size
    boxes: list[tuple[int, int, int, int]] = []
    patches: list[Tensor] = []
    for top in range(0, max(1, height - tile_size + 1), stride):
        for left in range(0, max(1, width - tile_size + 1), stride):
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)
            left = max(0, right - tile_size)
            top = max(0, bottom - tile_size)
            box = (left, top, right, bottom)
            boxes.append(box)
            patches.append(_image_to_tensor(image.crop(box), transform))

    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(patches), batch_size):
            batch = torch.stack(patches[start : start + batch_size]).to(
                device, non_blocking=True
            )
            probabilities.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(probabilities), np.asarray(boxes, dtype=np.int32)


def triage_from_probabilities(
    probabilities: np.ndarray,
    review_threshold: float = 0.10,
    reject_threshold: float = 0.30,
    min_confidence: float = 0.55,
) -> dict[str, Any]:
    """Aggregate probabilities into a conservative, confidence-aware report."""

    if probabilities.ndim != 2 or probabilities.shape[1] < max(ARTIFACT_CLASS_IDS) + 1:
        raise ValueError("probabilities must have shape (tiles, classes)")
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    artifact_probability = probabilities[:, ARTIFACT_CLASS_IDS].sum(axis=1)
    uncertain_fraction = float(np.mean(confidence < min_confidence))
    artifact_fraction = float(np.mean(artifact_probability >= 0.5))
    weighted_artifact_fraction = float(np.mean(artifact_probability))

    if uncertain_fraction > 0.25:
        verdict = "REVIEW: LOW CONFIDENCE"
    elif weighted_artifact_fraction >= reject_threshold:
        verdict = "SLIDE REJECTED"
    elif weighted_artifact_fraction >= review_threshold:
        verdict = "SLIDE: REVIEW RECOMMENDED"
    else:
        verdict = "SLIDE APPROVED"
    return {
        "verdict": verdict,
        "artifact_fraction": artifact_fraction,
        "weighted_artifact_fraction": weighted_artifact_fraction,
        "uncertain_fraction": uncertain_fraction,
        "mean_confidence": float(confidence.mean()),
        "predicted_class_counts": {
            CLASS_NAMES[index] if index < len(CLASS_NAMES) else str(index): int(np.sum(predicted == index))
            for index in np.unique(predicted)
        },
    }


def save_config(config: QCConfig, path: str | Path) -> None:
    """Persist configuration next to experiment artifacts."""

    Path(path).write_text(json.dumps(asdict(config), indent=2))
