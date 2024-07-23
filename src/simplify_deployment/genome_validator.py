from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from simplify_deployment.config_utils import Valid_filter, Valid_transformation


class Base_fields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variable: str
    lag: int

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "variable": [self.variable],
                "lag": [self.lag],
                "selected": [True],
            }
        )


class Transfo_fields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variable: str
    lag: int
    transformation: Valid_transformation

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "variable": [self.variable],
                "transformation": [self.transformation],
                "lag": [self.lag],
                "selected": [True],
            }
        )


class Filter_fields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variable: str
    lag: int
    filter: Valid_filter
    filter_period_hours: float

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "variable": [self.variable],
                "filter": [self.filter],
                "filter_period_hours": [self.filter_period_hours],
                "lag": [self.lag],
                "selected": [True],
            }
        )


class Genome_validator(BaseModel):
    model_config = ConfigDict(extra="forbid")
    genome_minute_base: Optional[list[Base_fields]] = None
    genome_minute_transfo: Optional[list[Transfo_fields]] = None
    genome_minute_filter: Optional[list[Filter_fields]] = None
    genome_quarter_base: Optional[list[Base_fields]] = None
    genome_quarter_transfo: Optional[list[Transfo_fields]] = None
    genome_quarter_filter: Optional[list[Filter_fields]] = None

    @classmethod
    def load(cls, path_genome: Path) -> "Genome_validator":
        with open(path_genome, "r") as f:
            unvalidated = yaml.safe_load(f)
        validated = Genome_validator(**unvalidated)
        return validated

    @property
    def df_genome_minute_base(self) -> pd.DataFrame:
        if self.genome_minute_base is None:
            return pd.DataFrame()
        else:
            return pd.concat(
                [item.df for item in self.genome_minute_base],
                axis=0,
            )

    @property
    def df_genome_minute_transfo(self) -> pd.DataFrame:
        if self.genome_minute_transfo is None:
            return pd.DataFrame
        else:
            return pd.concat(
                [item.df for item in self.genome_minute_transfo],
                axis=0,
            )
