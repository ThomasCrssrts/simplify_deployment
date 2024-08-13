import logging
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from simplify_deployment.data_wrangling import create_target, create_X
from simplify_deployment.organism import Organism

# Create logger
logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)

# Create handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Add handler to logger
logger.addHandler(handler)


def main(
    path_minute_data: Path,
    path_qh_data: Path,
    path_config_s1: Path,
    path_genome_s1: Path,
    path_output: Path,
):
    X = create_X(
        path_minute_data=path_minute_data,
        path_qh_data=path_qh_data,
    )

    X = X.asfreq("1min").ffill()
    tscv = TimeSeriesSplit(
        n_splits=12,
        max_train_size=16 * 7 * 24 * 60,
        gap=0,
        test_size=4 * 7 * 24 * 60,
    )
    prediction_list = []
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        logger.info(f"Starting fold {fold}.")
        # org
        org = Organism.from_yaml(
            path_config_s1,
            path_genome_s1,
        )
        # Train
        X_train = X.iloc[train_index]
        y_train = create_target(
            X_train,
        )

        # Train S1
        y_train_s1, X_train_s1 = org.create_y_X(
            y_train,
            X_train,
        )

        # Test
        X_test = X.iloc[test_index]
        y_test = create_target(
            X_test,
        )

        # Test S1
        y_test_s1, X_test_s1 = org.create_y_X(
            y_test,
            X_test,
        )

        # Model training
        model = LinearRegression()
        model.fit(X_train_s1, y_train_s1)
        prediction = pd.DataFrame(
            {"y_true": y_test, "y_pred_s1": model.predict(X_test_s1)},
            index=y_test_s1.index,
        )
        prediction_list.append(
            prediction,
        )
        logger.info(f"Fold {fold} done.")
    all_predictions = pd.concat(
        prediction_list,
        axis=0,
    )
    all_predictions.to_parquet(
        path_output,
    )


if __name__ == "__main__":
    main(
        Path("data/simplify_1_0/s1_minute_data.parquet"),
        Path("data/simplify_1_0/s1_quarter_data.parquet"),
        Path("src/simplify_deployment/config/lag_25.yaml"),
        Path("src/simplify_deployment/genomes/lag_25_simplify_1_0.yaml"),
        Path("data/simplify_1_0/predictions/s1_predictions.parquet"),
    )
