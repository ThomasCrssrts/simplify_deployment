import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import typer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Create logger
logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)

# Create handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Add handler to logger
logger.addHandler(handler)


class Model(nn.Module):
    def __init__(
        self,
        n_features: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=n_features,
            out_features=1,
        )  # Just 1 fully connected layer without activation,
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
        sigmoid_steepness: float = 1,
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
        loss_percentage_of_time_wrong_sign = (
            1 / (1 + torch.e ** (-self.steepness * (inputs * targets)))
        ).mean()

        # Total loss
        total_loss = (
            self.weight_max_error * max_error
            + self.weight_percentage_above_threshold
            * percentage_of_time_above_x
            + self.weight_wrong_sign * loss_percentage_of_time_wrong_sign
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


def main(
    path_to_X_train: Path = typer.Option(
        default=...,
    ),
    path_to_y_train: Path = typer.Option(
        default=...,
    ),
    path_to_X_test: Path = typer.Option(
        default=...,
    ),
    path_to_y_test: Path = typer.Option(
        default=...,
    ),
    path_to_save_predictions: Path = typer.Option(
        default=...,
    ),
):
    # Read in data
    X_train = pd.read_parquet(
        path_to_X_train,
    )
    y_train = pd.read_parquet(
        path_to_y_train,
    )
    X_test = pd.read_parquet(
        path_to_X_test,
    )
    y_test = pd.read_parquet(path_to_y_test)

    # Standardscale the data
    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    X_train_scaled = torch.Tensor(
        X_scaler.transform(X_train),
    ).float()

    X_test_scaled = torch.Tensor(
        X_scaler.transform(X_test),
    ).float()

    y_train_scaled = torch.Tensor(
        y_scaler.transform(
            y_train
        ).squeeze(),  # Make target tensor unidimensional
    ).float()

    # Init variables for training loop
    epochs = 5
    lr = 1e-4
    batch_size = 10
    n_features = X_train.shape[1]

    dataloader = DataLoader(
        CustomDataset(X_train_scaled, y_train_scaled),
        batch_size=batch_size,
        shuffle=True,
    )
    model = Model(
        n_features=n_features,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    threshold = 117
    converted_threshold = threshold / np.std(y_train, axis=0)
    converted_threshold = converted_threshold.item()
    criterion = CustomLoss(
        threshold=converted_threshold,
        weight_wrong_sign=0,
        weight_max_error=1,
        weight_percentage_above_threshold=0,
    )
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            prediction = model(X_batch)
            optimizer.zero_grad()
            loss = criterion(prediction, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        average_loss = epoch_loss / len(dataloader)
        logger.info(f"Average epoch loss: {average_loss}")
        logger.info(f"Epoch {epoch} done.")

    # Make final prediction on test
    model.eval()
    prediction = model(X_test_scaled).detach().numpy()
    prediction_unscaled = y_scaler.inverse_transform(
        prediction.reshape(-1, 1),
    )
    prediction_df = y_test.copy()
    prediction_df["y_pred"] = prediction_unscaled.squeeze()
    prediction_df.to_parquet(
        path_to_save_predictions,
    )
    logger.info("Finished.")


if __name__ == "__main__":
    typer.run(main)
