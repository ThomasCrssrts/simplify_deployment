import logging
from datetime import datetime
from pathlib import Path

import typer
from dateutil import tz
from sklearn.model_selection import TimeSeriesSplit

from simplify_deployment.data_wrangling import create_target, create_X

# Create logger
logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)

# Create handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Add handler to logger
logger.addHandler(handler)


def main(
    path_minute_data: Path = typer.Option(default=...),
    path_qh_data: Path = typer.Option(default=...),
    path_to_save_folds: Path = typer.Option(default=...),
):
    X_minute = create_X(
        path_minute_data=path_minute_data,
        path_qh_data=path_qh_data,
    )
    # load data untill 22 nov 2022 UTC. Later data has gaps.
    X_minute = (
        X_minute.loc[
            : datetime(2022, 11, 22, 0, 0, 0, tzinfo=tz.UTC), :  # type: ignore
        ]
        .asfreq("1min")
        .ffill()
    )

    tscv = TimeSeriesSplit(
        n_splits=12,
        max_train_size=16 * 7 * 24 * 60,
        gap=0,
        test_size=4 * 7 * 24 * 60,
    )

    for fold, (train_index, test_index) in enumerate(tscv.split(X_minute)):
        logger.info(f"Starting fold {fold}.")

        # Train
        X_train = X_minute.iloc[train_index]
        X_train.to_parquet(
            path_to_save_folds / f"X_train_fold_{fold}.parquet",
        )

        y_train = create_target(
            X_train,
        )
        y_train.to_frame().to_parquet(
            path_to_save_folds / f"y_train_fold_{fold}.parquet",
        )

        # Test
        X_test = X_minute.iloc[test_index]
        X_test.to_parquet(
            path_to_save_folds / f"X_test_fold_{fold}.parquet",
        )

        y_test = create_target(
            X_test,
        )
        y_test.to_frame().to_parquet(
            path_to_save_folds / f"y_test_fold_{fold}.parquet",
        )
        logger.info(f"Fold {fold} done.")


if __name__ == "__main__":
    main(
        path_minute_data=Path(
            "data/lots_of_vars/minute_data.parquet",
        ),
        path_qh_data=Path(
            "data/lots_of_vars/quarter_data.parquet",
        ),
        path_to_save_folds=Path(
            "/home/thomas/repos/simplify_deployment/data/folds"
        ),
    )
    # typer.run(main)
