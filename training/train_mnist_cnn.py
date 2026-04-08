#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 2026


def set_random_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_dataset_path(requested_path: Path) -> Path:
    """Resolve the dataset path, with a small fallback for the local repo filename."""
    if requested_path.exists():
        return requested_path

    fallback_path = requested_path.parent / "mnist_train.csv"
    if fallback_path.exists():
        print(f"Dataset '{requested_path.name}' not found, using '{fallback_path.name}' instead.")
        return fallback_path

    raise FileNotFoundError(f"Dataset file not found: {requested_path}")


def load_mnist_csv(csv_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load MNIST CSV data, normalize pixels to [0, 1], and reshape to N x 1 x 28 x 28."""
    dataframe = pd.read_csv(csv_path)

    labels = dataframe.iloc[:, 0].to_numpy(dtype=np.int64)
    pixels = dataframe.iloc[:, 1:].to_numpy(dtype=np.float32)
    pixels = pixels / 255.0
    pixels = pixels.reshape(-1, 1, 28, 28)

    images_tensor = torch.from_numpy(pixels)
    labels_tensor = torch.from_numpy(labels)
    return images_tensor, labels_tensor


def split_train_validation(
    images: torch.Tensor, labels: torch.Tensor, validation_ratio: float, seed: int
) -> tuple[TensorDataset, TensorDataset]:
    """Create a reproducible train/validation split from the loaded tensors."""
    num_samples = images.shape[0]
    if num_samples == 0:
        raise ValueError("The dataset is empty.")

    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    validation_size = int(num_samples * validation_ratio)
    validation_size = max(1, validation_size)
    validation_size = min(validation_size, num_samples - 1)

    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    train_dataset = TensorDataset(images[train_indices], labels[train_indices])
    validation_dataset = TensorDataset(images[validation_indices], labels[validation_indices])
    return train_dataset, validation_dataset


class SmallMNISTCNN(nn.Module):
    """A small CNN whose first conv layer matches the accelerator's first layer exactly."""

    def __init__(self) -> None:
        super().__init__()
        # First convolution layer matches the simulator configuration:
        # 1 input channel, 8 output channels, 3x3 kernel, stride 1, padding 1, with bias.
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(8 * 28 * 28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy for one pass over a dataloader."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))

    return correct / total if total > 0 else 0.0


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch over the training split."""
    model.train()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

    return evaluate(model, dataloader, device)


def export_conv1_parameters(model: SmallMNISTCNN, output_dir: Path) -> None:
    """Export conv1 weights and bias in a plain text format that C++ can parse easily."""
    weight_path = output_dir / "conv1_weight.txt"
    bias_path = output_dir / "conv1_bias.txt"

    conv_weight = model.conv1.weight.detach().cpu().numpy()
    conv_bias = model.conv1.bias.detach().cpu().numpy()

    with weight_path.open("w", encoding="utf-8") as weight_file:
        for oc in range(conv_weight.shape[0]):
            for cin in range(conv_weight.shape[1]):
                for ky in range(conv_weight.shape[2]):
                    for kx in range(conv_weight.shape[3]):
                        value = float(conv_weight[oc, cin, ky, kx])
                        weight_file.write(f"{oc} {cin} {ky} {kx} {value:.10f}\n")

    with bias_path.open("w", encoding="utf-8") as bias_file:
        for oc in range(conv_bias.shape[0]):
            value = float(conv_bias[oc])
            bias_file.write(f"{oc} {value:.10f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small MNIST CNN and export conv1 parameters.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "mnist_training.csv",
        help="Path to the MNIST CSV file.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples used for validation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "mnist_cnn_checkpoint.pt",
        help="Optional model checkpoint path.",
    )
    args = parser.parse_args()

    set_random_seeds(SEED)

    script_dir = Path(__file__).resolve().parent
    csv_path = resolve_dataset_path(args.csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the CSV dataset: first column is the label and the next 784 columns are image pixels.
    images, labels = load_mnist_csv(csv_path)
    train_dataset, validation_dataset = split_train_validation(
        images, labels, validation_ratio=args.validation_ratio, seed=SEED
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    # Build a simple network whose first conv layer exactly matches the accelerator's first layer.
    model = SmallMNISTCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Standard supervised training loop with per-epoch train and validation accuracy reporting.
    for epoch in range(1, args.epochs + 1):
        train_accuracy = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        validation_accuracy = evaluate(model, validation_loader, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_acc={train_accuracy * 100.0:.2f}% | "
            f"val_acc={validation_accuracy * 100.0:.2f}%"
        )

    # Save the full model checkpoint for later reuse if needed.
    torch.save(model.state_dict(), args.checkpoint)

    # Export only conv1 in a line-oriented text format:
    # weight lines: "oc cin ky kx value"
    # bias lines:   "oc value"
    export_conv1_parameters(model, script_dir)
    print(f"Saved checkpoint to: {args.checkpoint}")
    print(f"Exported conv1 weights to: {script_dir / 'conv1_weight.txt'}")
    print(f"Exported conv1 bias to: {script_dir / 'conv1_bias.txt'}")


if __name__ == "__main__":
    main()
