import logging
from pathlib import Path

import pandas as pd
import typer
from sklearn.linear_model import LinearRegression

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
    path_to_folds: Path = typer.Option(default=...),
    fold: int = typer.Option(default=...),
    path_best_genome: Path = typer.Option(default=...),
    path_to_save_predictions: Path = typer.Option(default=...),
    extra_organisms: list[Path] = typer.Option(default=...),
):

    X_train = pd.read_parquet(
        path_to_folds / f"X_train_fold_{fold}.parquet",
    )
    X_test = pd.read_parquet(
        path_to_folds / f"X_test_fold_{fold}.parquet",
    )

    y_train = pd.read_parquet(
        path_to_folds / f"y_train_fold_{fold}.parquet",
    )
    y_test = pd.read_parquet(path_to_folds / f"y_test_fold_{fold}.parquet")

    # Initialize org to determine amount of vars needed
    org = Organism(path_config)
    org._init_empty_genome()
    n_vars = org.get_n_variables_possible()

    # Find best org
    best_organism = genetic_algorithm(
        path_config=path_config,
        chance_of_random_variable_to_be_in_organism=1 / n_vars,
        mutation_chance=1 / n_vars,
        n_generations=100,
        n_untouched=1,
        number_of_deaths=50,
        population_size=200,
        reproduction_chance_second_over_first=0.90,
        X=X_train,
        y=y_train,
        extra_organisms=extra_organisms,
        path_best_genome=path_best_genome,
    )

    # Train model on variables
    model = LinearRegression()
    y_train_model, X_train_model = best_organism.create_y_X(
        y_train,
        X_train,
    )
    # For testing we take everything as some parts will be cut of by the lags.
    # This avoid gaps and later we filter the correct ones
    y_test_model, X_test_model = best_organism.create_y_X(
        pd.concat(
            [
                y_train,
                y_test,
            ],
            axis=0,
        ),
        pd.concat(
            [
                X_train,
                X_test,
            ],
            axis=0,
        ),
    )
    model.fit(
        X_train_model,
        y_train_model,
    )
    predictions = pd.DataFrame(
        {
            "y_pred": model.predict(X_test_model),
            "y_true": y_test_model.values,
        },
        index=y_test_model.index,
    )
    mask = [x for x in y_test.index if x in predictions.index]
    predictions = predictions.loc[mask, :]
    predictions.to_parquet(path_to_save_predictions)


if __name__ == "__main__":
    typer.run(main)
