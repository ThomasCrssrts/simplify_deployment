from enum import StrEnum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel
from scipy.signal import butter, sosfilt


class Valid_transformation(StrEnum):
    first_derivative = "first_derivative"
    second_derivative = "second_derivative"
    cumsum_during_quarter = "cumsum_during_quarter"

    def apply(
        self,
        X: pd.Series,
    ) -> pd.Series:
        match self:
            case Valid_transformation.first_derivative:
                transformed = (
                    X.diff(1)
                    .fillna(0)
                    .rename(
                        "{}_first_derivative".format(
                            X.name,
                        )
                    )
                )
                return transformed
            case Valid_transformation.second_derivative:
                transformed = (
                    X.diff(1)
                    .diff(1)
                    .fillna(0)
                    .rename(
                        "{}_second_derivative".format(
                            X.name,
                        )
                    )
                )
                return transformed
            case Valid_transformation.cumsum_during_quarter:
                transformed = (
                    X.groupby(X.index.floor("15min"))
                    .cumsum()
                    .rename(
                        "{}_cumsum_during_quarter".format(
                            X.name,
                        )
                    )
                )
                return transformed
            case _:
                raise ValueError("Unknown transformation")


class Valid_filter(StrEnum):
    low_pass = "low_pass"
    band_pass = "band_pass"
    high_pass = "high_pass"


class Filter(BaseModel):
    kind: Valid_filter
    period_hours: float

    @property
    def sos_filter(self) -> np.array:
        match self.kind:
            case Valid_filter.low_pass:
                return butter(
                    N=2,
                    Wn=[1 / (self.period_hours * 60)],
                    btype="lp",
                    fs=1,
                    output="sos",
                )
            case Valid_filter.high_pass:
                return butter(
                    N=2,
                    Wn=[1 / (self.period_hours * 60)],
                    btype="hp",
                    fs=1,
                    output="sos",
                )
            case Valid_filter.band_pass:
                return butter(
                    N=2,
                    Wn=[
                        1 / (self.period_hours * 60 + 5),
                        1 / (self.period_hours * 60 - 5),
                    ],
                    btype="bp",
                    fs=1,
                    output="sos",
                )
            case _:
                raise ValueError("Unknown filter type.")

    def apply(
        self,
        X: pd.Series,
    ) -> pd.Series:
        sos = self.sos_filter
        var = X.name
        filtered = pd.Series(
            data=sosfilt(
                sos,
                X.values,
            ),
            name="{}_{}_{}_h".format(
                var,
                self.kind.value,
                str(self.period_hours).replace(".", "_"),
            ),
            index=X.index,
        )
        return filtered


class Variable_fields(BaseModel):
    min_lag: int
    max_lag: int
    transformations: Optional[list[Valid_transformation]] = None
    filters: Optional[list[Filter]] = None


class Config(BaseModel):
    minute: Optional[dict[str, Variable_fields]] = None
    quarter: Optional[dict[str, Variable_fields]] = None

    @classmethod
    def load(cls, path_config: Path) -> "Config":
        with open(path_config, "r") as f:
            unvalidated = yaml.safe_load(f)
        validated = Config(**unvalidated)
        return validated
