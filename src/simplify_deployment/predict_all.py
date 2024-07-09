import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import typer
from dateutil import tz
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from simplify_deployment.data_wrangling import create_target, create_X
from simplify_deployment.genetic_algorithm import genetic_algorithm
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
    path_config: Path = typer.Option(
        default=...,
    ),
    path_minute_data: Path = typer.Option(default=...),
    path_qh_data: Path = typer.Option(default=...),
    path_best_genome: Path = typer.Option(default=...),
    path_to_save_predictions: Path = typer.Option(default=...),
    extra_organisms: list[Path] = typer.Option(default=...),
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
    prediction_list = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X_minute)):
        logger.info(f"Starting fold {fold}.")

        # Train
        X_train = X_minute.iloc[train_index]
        y_train = create_target(
            X_train,
        )

        # Test
        X_test = X_minute.iloc[test_index]
        y_test = create_target(
            X_test,
        )
        # Check how many variables are possible
        # This will help set some probabilities
        org = Organism(path_config)
        org._init_empty_genome()
        n_vars = org.get_n_variables_possible()
        # Find best organism
        if fold == 0:
            best_organism = genetic_algorithm(
                path_config=path_config,
                chance_of_random_variable_to_be_in_organism=1 / n_vars,
                mutation_chance=1 / n_vars,
                n_generations=50,
                n_untouched=1,
                number_of_deaths=10,
                population_size=200,
                reproduction_chance_second_over_first=0.85,
                X=X_train,
                y=y_train,
                extra_organisms=extra_organisms,
                path_best_genome=path_best_genome,
            )
        else:
            best_organism = genetic_algorithm(
                path_config=path_config,
                chance_of_random_variable_to_be_in_organism=1 / n_vars,
                mutation_chance=1 / n_vars,
                n_generations=25,
                n_untouched=1,
                number_of_deaths=10,
                population_size=200,
                reproduction_chance_second_over_first=0.85,
                X=X_train,
                y=y_train,
                extra_organisms=extra_organisms + [path_best_genome],
                path_best_genome=path_best_genome,
            )
        # Train model on variables
        model = LinearRegression()
        y_train_model, X_train_model = best_organism.create_y_X(
            y_train,
            X_train,
        )
        y_test_model, X_test_model = best_organism.create_y_X(
            y_test,
            X_test,
        )
        model.fit(
            X_train_model,
            y_train_model,
        )
        prediction_list.append(
            pd.DataFrame(
                {
                    "prediction": model.predict(X_test_model),
                    "real": y_test_model.values,
                },
                index=y_test_model.index,
            ),
        )
    all_predictions = pd.concat(
        prediction_list,
        axis=0,
    )
    all_predictions.to_parquet(Path(path_to_save_predictions))


if __name__ == "__main__":
    # main(
    #     path_config=Path("/home/thomas/repos/simplify_deployment/src/simplify_deployment/config/lag_25.yaml"),
    #     path_minute_data=Path("/home/thomas/repos/simplify_deployment/data/simplify_1_0/minute_data.parquet"),
    #     path_qh_data=Path("/home/thomas/repos/simplify_deployment/data/simplify_1_0/quarter_data.parquet"),
    #     path_best_genome=Path("/home/thomas/repos/simplify_deployment/src/simplify_deployment/genomes/lag_25_best_genome.yaml"),
    #     path_to_save_predictions=Path("/home/thomas/repos/simplify_deployment/data/predictions.parquet"),
    #     extra_organisms=[
    #                 Path(
    #                     "/home/thomas/repos/simplify_deployment/src/simplify_deployment/genomes/lag_25_simplify_1_0.yaml"
    #                 ),
    #                 Path(
    #                     "/home/thomas/repos/simplify_deployment/src/simplify_deployment/genomes/lag_25_gen_2.yaml"
    #                 ),
    #     ],
    # )
    typer.run(main)
