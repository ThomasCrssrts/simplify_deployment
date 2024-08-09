from pathlib import Path

import pandas as pd
import typer


def main(
    paths_predictions: list[Path] = typer.Option(
        default=...,
    ),
    path_output: Path = typer.Option(
        default=...,
    ),
):
    prediction_list = []
    for path_prediction in paths_predictions:
        prediction_list.append(
            pd.read_parquet(
                path_prediction,
            ),
        )
    prediction_df = pd.concat(
        prediction_list,
        axis=0,
    )
    prediction_df.to_parquet(path_output)


if __name__ == "__main__":
    typer.run(main)
