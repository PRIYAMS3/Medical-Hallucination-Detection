from __future__ import annotations

import torch
import torch.nn as nn


class SimpleANN(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DeepANN(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DropoutANN(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, 64)
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.output_layer = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(x))
        x = self.res1(x)
        x = self.res2(x)
        return self.output_layer(x)


class WideDeep(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.output = nn.Linear(input_dim + 32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deep_out = self.deep(x)
        combined = torch.cat([x, deep_out], dim=1)
        return self.output(combined)


class CNN1D(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class AttentionModel(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x)
        x = x * weights
        return self.fc(x)


MODEL_REGISTRY = {
    "simple_ann": SimpleANN,
    "deep_ann": DeepANN,
    "dropout_ann": DropoutANN,
    "residual_mlp": ResidualMLP,
    "wide_deep": WideDeep,
    "cnn_1d": CNN1D,
    "attention": AttentionModel,
}


def available_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def build_model(model_name: str, input_dim: int) -> nn.Module:
    key = model_name.strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {available_models()}")
    return MODEL_REGISTRY[key](input_dim)
