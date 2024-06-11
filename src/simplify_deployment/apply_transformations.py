import logging
from enum import StrEnum, auto
from pathlib import Path

import pandas as pd
import typer
import yaml
from pydantic import BaseModel, ConfigDict, field_validator

# Create logger
logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)

# Create handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Add handler to logger
logger.addHandler(handler)


class Valid_transformation_types(StrEnum):
    first_derivative = auto()
    second_derivative = auto()
    cumsum_during_quarter = auto()


class Transformation_schema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Valid_transformation_types
    columns: list[str]

    @field_validator("columns")
    @classmethod
    def validate_unique_columns(cls, v: list[str]) -> list[str]:
        return list(set(v))

    @property
    def name(self) -> str:
        return f"{self.kind}"

    def apply_transformation(self, X: pd.Series) -> pd.Series:
        match self.kind:
            case Valid_transformation_types.first_derivative:
                return X.diff().fillna(0).rename(f"{X.name}_first_derivative")
            case Valid_transformation_types.second_derivative:
                return (
                    X.diff()
                    .diff()
                    .fillna(0)
                    .rename(f"{X.name}_second_derivative")
                )
            case Valid_transformation_types.cumsum_during_quarter:
                return (
                    X.groupby(X.index.floor("15min"))
                    .cumsum()
                    .rename(f"{X.name}_cumsum_during_quarter")
                )


def main(
    path_to_config_transformations: Path = typer.Option(default=...),
    path_to_X: Path = typer.Option(default=...),
    path_output: Path = typer.Option(default=...),
):
    logger.info("Start reading transformation config.")
    # Read in the raw yaml. Gives list of filter schemas.
    with open(path_to_config_transformations, "r") as f:
        config_data = yaml.safe_load(f)

    # Validate transformation schemas.
    validated_config_data = [Transformation_schema(**x) for x in config_data]
    logger.info("Filter config validated.")

    # Read in X data
    logger.info("Reading in X data.")
    X = pd.read_parquet(path_to_X)

    # Now we apply each transformation to all the columns desired
    logger.info("Start applying filters.")
    transformed_columns_list = []
    for transformer in validated_config_data:
        for col in transformer.columns:
            try:
                transformed_columns_list.append(
                    transformer.apply_transformation(X[col])
                )
            except KeyError as e:
                logger.error(f"Column {e} not found in X.")
    transformed_columns = pd.concat(transformed_columns_list, axis=1)
    transformed_columns.to_parquet(path_output)
    logger.info("All transformations done.")


if __name__ == "__main__":
    typer.run(main)
