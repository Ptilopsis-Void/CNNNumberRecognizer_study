from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    """Compact CNN for 28 x 28 grayscale digit images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.pool(F.relu(self.conv1(inputs)))
        features = self.pool(F.relu(self.conv2(features)))
        features = self.pool(F.relu(self.conv3(features)))
        features = self.dropout1(features)
        features = torch.flatten(features, start_dim=1)
        features = self.dropout2(F.relu(self.fc1(features)))
        return self.fc2(features)


def create_model(num_classes: int = 10) -> CNN:
    return CNN(num_classes=num_classes)

