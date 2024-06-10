import logging
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
    path_to_first_df: Path = typer.Option(default=...),
    path_to_second_df: Path = typer.Option(default=...),
    path_to_save_merged_df: Path = typer.Option(default=...),
) -> None:
    logger.info("Starting merging")
    (
        pd.merge(
            pd.read_parquet(path_to_first_df),
            pd.read_parquet(path_to_second_df),
            left_index=True,
            right_index=True,
            how="inner",
        ).to_parquet(path_to_save_merged_df)
    )
    logger.info("Merging done.")
    return


if __name__ == "__main__":
    typer.run(main)
