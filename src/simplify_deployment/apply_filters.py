import logging
from enum import StrEnum, auto
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml
from pydantic import BaseModel, ConfigDict, field_validator
from scipy.signal import butter, sosfilt

# Create logger
logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)

# Create handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Add handler to logger
logger.addHandler(handler)


class Valid_filter_types(StrEnum):
    low_pass = auto()
    band_pass = auto()
    high_pass = auto()


class Filter_schema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Valid_filter_types
    period_hours: float
    columns: list[str]

    @field_validator("period_hours")
    @classmethod
    def validate_period_bigger_than_nyquist(cls, v: float) -> float:
        # To make mathematical sense the filter period needs
        # to be bigger than or equal than sampling period x 2
        # Sampling period in hours is 1/60
        if v <= 2 / 60:
            raise ValueError(
                "The desired filter period should be bigger\
                      than the nyquist period of 2/60 hour (2minutes)."
            )
        else:
            return v

    @field_validator("columns")
    @classmethod
    def validate_unique_columns(cls, v: list[str]) -> list[str]:
        return list(set(v))

    @property
    def name(self) -> str:
        return f"{self.kind}_{self.period_hours}_hours".replace(".", "_")

    def get_butter(self) -> np.array:
        match self.kind:
            case Valid_filter_types.band_pass:
                lower_freq = 1 / (self.period_hours * 60 + 5)
                upper_freq = 1 / (self.period_hours * 60 - 5)
                return butter(
                    N=2,
                    Wn=[lower_freq, upper_freq],
                    btype="bp",
                    fs=1,
                    output="sos",
                )
            case Valid_filter_types.low_pass:
                freq = 1 / (self.period_hours * 60)
                return butter(
                    N=2,
                    Wn=[freq],
                    btype="lp",
                    fs=1,
                    output="sos",
                )
            case Valid_filter_types.high_pass:
                freq = 1 / (self.period_hours * 60)
                return butter(
                    N=2,
                    Wn=[freq],
                    btype="hp",
                    fs=1,
                    output="sos",
                )
            case _:
                raise ValueError("Unknown filter type")


def main(
    path_to_config_filters: Path = typer.Option(default=...),
    path_to_X: Path = typer.Option(default=...),
    path_output: Path = typer.Option(default=...),
):
    logger.info("Start reading filter config.")
    # Read in the raw yaml. Gives list of filter schemas.
    with open(path_to_config_filters, "r") as f:
        config_data = yaml.safe_load(f)

    # Validate filter schemas.
    validated_config_data = [Filter_schema(**x) for x in config_data]
    logger.info("Filter config validated.")

    # Read in X data
    logger.info("Reading in X data.")
    X = pd.read_parquet(path_to_X)

    # Now we apply each filter to all the columns desired
    logger.info("Start applying filters.")
    filtered_columns_list = []
    for filter in validated_config_data:
        for col in filter.columns:
            try:
                filtered_columns_list.append(
                    pd.DataFrame(
                        sosfilt(filter.get_butter(), X[col]),
                        index=X[col].index,
                        columns=[f"{col}_{filter.name}"],
                    )
                )
            except KeyError as e:
                logger.error(f"Column {e} not found in X.")
    filtered_columns = pd.concat(filtered_columns_list, axis=1)
    filtered_columns.to_parquet(path_output)
    logger.info("All filters done.")


if __name__ == "__main__":
    typer.run(main)
