import logging
from functools import reduce
from pathlib import Path

import pandas as pd
import typer

# Create logger
logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)

# Create handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Add handler to logger
logger.addHandler(handler)


def main(
    paths: list[Path] = typer.Option(..., help="Input df paths."),
    path_output: Path = typer.Option(..., help="Path to output df"),
) -> None:
    dfs = [pd.read_parquet(df_path) for df_path in paths]
    (
        reduce(
            lambda a, b: pd.merge(
                a,
                b,
                left_index=True,
                right_index=True,
                how="inner",
            ),
            dfs,
        ).to_parquet(path_output)
    )


if __name__ == "__main__":
    typer.run(main)
