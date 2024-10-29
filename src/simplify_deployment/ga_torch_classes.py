from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Model(nn.Module):
    def __init__(
        self,
        n_features: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=n_features,
            out_features=1,
        )
        # Just 1 fully connected layer without activation,
        # i.e. a linear regression.

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        y = self.fc1(X)
        return y.flatten()


class CustomLoss(nn.Module):
    def __init__(
        self,
        threshold: float = 0.685,
        weight_max_error: float = 1,
        weight_percentage_above_threshold: float = 1,
        weight_wrong_sign: float = 1,
        sigmoid_steepness: float = 100,
    ) -> None:
        super().__init__()
        self.steepness = sigmoid_steepness
        self.threshold = threshold
        # Normalize weights and assign them
        sum_weights = (
            weight_max_error
            + weight_percentage_above_threshold
            + weight_wrong_sign
        )
        self.weight_max_error = weight_max_error / sum_weights
        self.weight_percentage_above_threshold = (
            weight_percentage_above_threshold / sum_weights
        )
        self.weight_wrong_sign = weight_wrong_sign / sum_weights

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        residuals = targets - inputs
        # Maximum abs error
        max_error = residuals.abs().max()

        # Percentage of time above threshold value
        percentage_of_time_above_x = (
            1
            / (
                1
                + torch.e
                ** (-self.steepness * (residuals.abs() - self.threshold))
            )
        ).mean()

        # Percentage of time wrong sign
        wrong_sign = (
            1 / (1 + torch.e ** (-self.steepness * (inputs * targets)))
        ).mean()

        # Total loss
        total_loss = (
            self.weight_max_error * max_error
            + self.weight_percentage_above_threshold
            * percentage_of_time_above_x
            + self.weight_wrong_sign * wrong_sign
        )
        return total_loss


class CustomDataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X_item = self.X[idx, :]
        y_item = self.y[idx]
        return X_item, y_item
