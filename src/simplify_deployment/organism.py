from datetime import timedelta
from functools import reduce
from itertools import product
from pathlib import Path
from typing import Literal, Tuple, Type, TypeVar

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score

from simplify_deployment.config_utils import (
    Config,
    Filter,
    Valid_transformation,
)
from simplify_deployment.genome_validator import Genome_validator

T = TypeVar("T", bound="Organism")


class Organism:
    def __init__(self, path_config: Path) -> None:
        self.fitness = -np.inf
        self.path_config = path_config

    def _create_genome_transfo(
        self,
        config: Config,
        granularity: Literal["minute", "qh"],
    ) -> pd.DataFrame:
        transformation_dfs = []
        # Minute data transformations
        if granularity == "minute":
            if not (config.minute is None):
                config_items = config.minute.items()
                step = 1
            else:
                return pd.DataFrame
        elif granularity == "qh":
            if not (config.quarter is None):
                config_items = config.quarter.items()
                step = 15
            else:
                return pd.DataFrame()
        for key, value in config_items:
            lags = range(value.min_lag, value.max_lag + 1, step)
            if not (value.transformations is None):
                transformations = [item.value for item in value.transformations]
                prod = product(
                    [key],
                    transformations,
                    lags,
                )
                prod_zip = list(zip(*prod))
                transformation_dfs.append(
                    pd.DataFrame(
                        {
                            "variable": prod_zip[0],
                            "transformation": prod_zip[1],
                            "lag": prod_zip[2],
                        }
                    )
                )
        if transformation_dfs == []:
            return pd.DataFrame()
        else:
            transformation_df = pd.concat(
                transformation_dfs,
                axis=0,
            )
            return transformation_df

    def _create_genome_filter(
        self,
        config: Config,
        granularity: Literal["minute", "qh"],
    ) -> pd.DataFrame:
        filters_dfs = []
        if granularity == "minute":
            if not (config.minute is None):
                config_items = config.minute.items()
                step = 1
            else:
                return pd.DataFrame()
        elif granularity == "qh":
            if not (config.quarter is None):
                config_items = config.quarter.items()
                step = 15
            else:
                return pd.DataFrame()
        for key, value in config_items:
            lags = range(value.min_lag, value.max_lag + 1, step)
            if not (value.filters is None):
                filters = value.filters
                prod = product(
                    [key],
                    filters,
                    lags,
                )
                prod_zip = list(zip(*prod))
                filters_dfs.append(
                    pd.DataFrame(
                        {
                            "variable": prod_zip[0],
                            "filter": [x.kind.value for x in prod_zip[1]],
                            "filter_period_hours": [
                                x.period_hours for x in prod_zip[1]
                            ],
                            "lag": prod_zip[2],
                        }
                    )
                )
        if filters_dfs == []:
            return pd.DataFrame()
        else:
            filters_df = pd.concat(
                filters_dfs,
                axis=0,
            )
            return filters_df

    def _create_genome_base(
        self,
        config: Config,
        granularity: Literal["minute", "qh"],
    ) -> pd.DataFrame:
        base_dfs = []
        if granularity == "minute":
            if not (config.minute is None):
                config_items = config.minute.items()
                step = 1
            else:
                return pd.DataFrame()
        elif granularity == "qh":
            if not (config.quarter is None):
                config_items = config.quarter.items()
                step = 15
            else:
                return pd.DataFrame()
        for key, value in config_items:
            lags = range(value.min_lag, value.max_lag + 1, step)
            prod = product(
                [key],
                lags,
            )
            prod_zip = list(zip(*prod))
            base_dfs.append(
                pd.DataFrame(
                    {
                        "variable": prod_zip[0],
                        "lag": prod_zip[1],
                    }
                )
            )
        if base_dfs == []:
            return pd.DataFrame()
        else:
            base_df = pd.concat(
                base_dfs,
                axis=0,
            )
            return base_df

    def init_random_genome(
        self,
        chance_to_select_variable: float = 0.1,
    ) -> None:
        # Read in config
        config = Config.load(self.path_config)

        # Minute base genome
        self.genome_minute_base = self._create_genome_base(
            config=config,
            granularity="minute",
        )

        # Minute filters genome
        self.genome_minute_filter = self._create_genome_filter(
            config=config,
            granularity="minute",
        )

        # Minute transformations genome
        self.genome_minute_transfo = self._create_genome_transfo(
            config=config,
            granularity="minute",
        )

        # Quarter base genome
        self.genome_quarter_base = self._create_genome_base(
            config=config,
            granularity="qh",
        )

        # Quarter filters genome
        self.genome_quarter_filter = self._create_genome_filter(
            config=config,
            granularity="qh",
        )

        # Minute transformations genome
        self.genome_quarter_transfo = self._create_genome_transfo(
            config=config,
            granularity="qh",
        )

        # Now select rows randomly
        for item in [
            self.genome_minute_base,
            self.genome_minute_filter,
            self.genome_minute_transfo,
            self.genome_quarter_base,
            self.genome_quarter_filter,
            self.genome_quarter_transfo,
        ]:
            item["selected"] = np.random.choice(
                a=[True, False],
                replace=True,
                p=[chance_to_select_variable, 1 - chance_to_select_variable],
                size=item.shape[0],
            )
        return

    def _init_empty_genome(
        self,
    ) -> None:
        self.init_random_genome(0)
        return

    def _df_from_base_genomes(
        self,
        X_minute,
    ) -> pd.DataFrame:
        series_list = []
        non_empty = [
            x
            for x in [
                self.genome_minute_base,
                self.genome_quarter_base,
            ]
            if not x.empty
        ]
        for item in non_empty:
            selected = item.loc[item["selected"], :]
            if not (selected.empty):
                for index, row in selected.iterrows():
                    series_list.append(
                        (
                            X_minute.loc[:, row["variable"]]
                            .shift(freq=timedelta(minutes=row["lag"]))
                            .rename(f'{row["variable"]}_lag_{row["lag"]}')
                        )
                    )
        if not (series_list == []):
            df = reduce(
                lambda a, b: pd.merge(
                    a,
                    b,
                    left_index=True,
                    right_index=True,
                ),
                series_list,
            )
            return df
        else:
            return pd.DataFrame()

    def _df_from_filter_genomes(
        self,
        X_minute: pd.DataFrame,
    ) -> pd.DataFrame:
        series_list = []
        non_empty = [
            x
            for x in [
                self.genome_minute_filter,
                self.genome_quarter_filter,
            ]
            if not x.empty
        ]
        for item in non_empty:
            selected = item.loc[item["selected"], :]
            if not (selected.empty):
                for index, row in selected.iterrows():
                    filter = Filter(
                        kind=row["filter"],
                        period_hours=row["filter_period_hours"],
                    )
                    filter.sos_filter
                    var = row["variable"]
                    lag = row["lag"]
                    original = X_minute.loc[:, var]
                    filtered = filter.apply(original)
                    filtered = filtered.shift(
                        freq=timedelta(minutes=lag)
                    ).rename(
                        "{}_lag_{}".format(
                            filtered.name,
                            lag,
                        )
                    )
                    series_list.append(filtered)
        if not (series_list == []):
            df = reduce(
                lambda a, b: pd.merge(
                    a,
                    b,
                    left_index=True,
                    right_index=True,
                ),
                series_list,
            )
            return df
        else:
            return pd.DataFrame()

    def _df_from_transfo_genomes(
        self,
        X_minute: pd.DataFrame,
    ) -> pd.DataFrame:
        series_list = []
        non_empty = [
            x
            for x in [
                self.genome_minute_transfo,
                self.genome_quarter_transfo,
            ]
            if not x.empty
        ]
        for item in non_empty:
            selected = item.loc[item["selected"], :]
            if not (selected.empty):
                for index, row in selected.iterrows():
                    transfo = Valid_transformation(row["transformation"])
                    transformed = transfo.apply(
                        (X_minute.loc[:, row["variable"]])
                    )
                    transformed = transformed.shift(
                        freq=timedelta(minutes=row["lag"])
                    ).rename(
                        "{}_lag_{}".format(
                            transformed.name,
                            row["lag"],
                        )
                    )
                    series_list.append(transformed)
        if not (series_list == []):
            df = reduce(
                lambda a, b: pd.merge(
                    a,
                    b,
                    left_index=True,
                    right_index=True,
                ),
                series_list,
            )
            return df
        else:
            return pd.DataFrame()

    def create_y_X(
        self,
        y: pd.DataFrame,
        X_minute: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dfs = [
            self._df_from_base_genomes(X_minute),
            self._df_from_filter_genomes(X_minute),
            self._df_from_transfo_genomes(X_minute),
        ]
        non_empty_dfs = [x for x in dfs if not (x.empty)]
        if not (non_empty_dfs == []):
            non_empty_dfs = [y] + non_empty_dfs
            y_X = reduce(
                lambda a, b: pd.merge(
                    a,
                    b,
                    left_index=True,
                    right_index=True,
                    how="inner",
                ),
                non_empty_dfs,
            )
            return y_X.iloc[:, 0], y_X.iloc[:, 1:]
        else:
            return pd.DataFrame(), pd.DataFrame()

    def calculate_fitness(
        self,
        y: pd.DataFrame,
        X_minute: pd.DataFrame,
    ) -> None:
        y_model, X_model = self.create_y_X(
            y,
            X_minute,
        )
        if y_model.empty:
            self.fitness = -np.inf
        else:
            model = LinearRegression()
            self.fitness = np.mean(
                cross_val_score(
                    model,
                    X_model,
                    y_model,
                    scoring="neg_root_mean_squared_error",
                    cv=ShuffleSplit(3, test_size=0.25),
                )
            )

    @classmethod
    def reproduce(
        cls: Type[T],
        first_organism: T,
        second_organism: T,
    ) -> T:
        # We make a random mask.
        # If mask is true, select gene from first organism.
        # If mask is false, select gene from second organism.
        new_org = cls(path_config=first_organism.path_config)
        for gen in [
            "genome_minute_base",
            "genome_minute_filter",
            "genome_minute_transfo",
            "genome_quarter_base",
            "genome_quarter_filter",
            "genome_quarter_transfo",
        ]:
            gen_first_org = getattr(
                first_organism,
                gen,
            )
            gen_second_org = getattr(
                second_organism,
                gen,
            )
            if not (gen_first_org.empty | gen_second_org.empty):
                gen_new = gen_first_org.copy()
                mask = np.random.choice(
                    a=[True, False],
                    p=[1 / 2, 1 / 2],
                    replace=True,
                    size=gen_new.shape[0],
                )
                gen_new["selected"] = (mask & gen_first_org["selected"]) | (
                    ~mask & gen_second_org["selected"]
                )
                setattr(
                    new_org,
                    gen,
                    gen_new,
                )
            else:
                setattr(
                    new_org,
                    gen,
                    pd.DataFrame(),
                )
        return new_org

    def mutate(
        self,
        mutation_chance: float = 0.05,
    ) -> None:

        def probability_func(
            df: pd.DataFrame,
            n_neighbors: int = 1,
        ) -> pd.DataFrame:
            if df.shape[0] == 1:
                df["chance"] = mutation_chance
                return df
            values_to_mutate = [
                df["selected"]
                .shift(
                    i,
                    fill_value=False,
                )
                .values
                for i in range(
                    -n_neighbors,
                    n_neighbors + 1,
                )
            ]
            selection = reduce(lambda a, b: a | b, values_to_mutate)
            df["chance"] = np.where(
                selection,
                10 * mutation_chance,
                mutation_chance,
            )
            return df

        for gen in [
            "genome_minute_base",
            "genome_minute_filter",
            "genome_minute_transfo",
            "genome_quarter_base",
            "genome_quarter_filter",
            "genome_quarter_transfo",
        ]:
            gen_old = getattr(
                self,
                gen,
            )
            if not (gen_old.empty):
                gen_new = gen_old.copy()
                sort_group = list(gen_new.drop(columns="selected").columns)
                probability_group = list(
                    gen_new.drop(columns=["selected", "lag"]).columns
                )
                probability_df = (
                    gen_new.sort_values(
                        by=sort_group,
                    )
                    .groupby(
                        probability_group,
                        group_keys=False,
                    )
                    .apply(probability_func)
                )
                random_numbers = np.random.rand(probability_df.shape[0])
                toggle = random_numbers < probability_df["chance"]
                # Mutation is just an xor with the toggle
                gen_new["selected"] = gen_new["selected"] ^ toggle
                self.genome = gen_new
                setattr(self, gen, gen_new)
            else:
                setattr(self, gen, pd.DataFrame())

    def get_n_variables_used(self) -> int:
        total = 0
        for gen in [
            "genome_minute_base",
            "genome_minute_filter",
            "genome_minute_transfo",
            "genome_quarter_base",
            "genome_quarter_filter",
            "genome_quarter_transfo",
        ]:
            df = getattr(
                self,
                gen,
            )
            if not (df.empty):
                total += df["selected"].astype(int).sum()
        return total

    def get_n_variables_possible(self) -> int:
        total = 0
        for gen in [
            "genome_minute_base",
            "genome_minute_filter",
            "genome_minute_transfo",
            "genome_quarter_base",
            "genome_quarter_filter",
            "genome_quarter_transfo",
        ]:
            df = getattr(
                self,
                gen,
            )
            if not (df.empty):
                total += df["selected"].shape[0]
        return total

    def get_variables_as_list_of_str(self) -> list[str]:
        text_list = []
        for gen in [
            "genome_minute_base",
            "genome_minute_filter",
            "genome_minute_transfo",
            "genome_quarter_base",
            "genome_quarter_filter",
            "genome_quarter_transfo",
        ]:
            df = getattr(
                self,
                gen,
            )
            if not (df.empty):
                df = df.loc[df["selected"], :]
                if not (df.empty):
                    text_list.append(f"{df}")
        return text_list

    def to_yaml(
        self,
        path_yaml: Path,
    ) -> None:
        data_to_dump = {}
        for gen in [
            "genome_minute_base",
            "genome_minute_filter",
            "genome_minute_transfo",
            "genome_quarter_base",
            "genome_quarter_filter",
            "genome_quarter_transfo",
        ]:
            df = getattr(
                self,
                gen,
            )
            if not (df.empty):
                df = df.loc[df["selected"], :].drop(columns="selected")
            if not (df.empty):
                data_to_dump[gen] = df.to_dict(
                    orient="records",
                )
        with open(path_yaml, "w") as f:
            yaml.dump(
                data_to_dump,
                f,
                indent=4,
                default_flow_style=False,
            )

    @classmethod
    def from_yaml(
        cls,
        path_config: Path,
        path_genome: Path,
    ) -> "Organism":
        validated = Genome_validator.load(path_genome=path_genome)
        org = Organism(path_config)
        org._init_empty_genome()
        for field in validated.model_fields_set:
            org_df = getattr(
                org,
                field,
            ).copy()
            validated_df = pd.concat(
                [x.df for x in getattr(validated, field)], axis=0
            )
            merged_df = pd.merge(
                org_df.drop(columns="selected"),
                validated_df.drop(columns="selected"),
                how="left",
                indicator=True,
            )
            mask = merged_df["_merge"] == "both"
            org_df["selected"] = mask.values
            setattr(org, field, org_df)
        return org

    def __lt__(self, other: "Organism") -> bool:
        if other.fitness > self.fitness:
            return True
        elif other.fitness == self.fitness:
            if other.get_n_variables_used() < self.get_n_variables_used():
                return True
            else:
                return False
        else:
            return False
