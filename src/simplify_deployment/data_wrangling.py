import logging
from pathlib import Path

import pandas as pd

# Create logger
logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)

# Create handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Add handler to logger
logger.addHandler(handler)


def create_target(
    minute_data: pd.DataFrame,
    target_name: str = "siCumulative",
) -> pd.DataFrame:
    logger.info("starting target creation")
    minute_data = minute_data.sort_index(ascending=True)
    target: pd.Series = minute_data.loc[:, target_name]
    target = target.loc[target.index.minute % 15 == 14]
    logger.info("Target created")
    return target


def resample_qh_data(
    qh_data: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Starting resampling of qh data.")
    qh_data = qh_data.resample(
        rule="1min",
        closed="left",
        label="left",
    ).ffill()
    logger.info("Resampling of qh data done.")
    return qh_data


def create_X(
    path_minute_data: Path,
    path_qh_data: Path,
):
    qh_data = pd.read_parquet(
        path_qh_data,
    )
    minute_data = pd.read_parquet(
        path_minute_data,
    )
    qh_data_resampled = resample_qh_data(
        qh_data,
    )
    X = pd.merge(
        minute_data, qh_data_resampled, left_index=True, right_index=True
    )
    return X
