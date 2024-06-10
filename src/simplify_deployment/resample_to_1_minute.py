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
    path_to_qh_data: Path = typer.Option(default=...),
    path_to_save_resampled_data: Path = typer.Option(default=...),
) -> None:
    logger.info("Starting resampling of qh data.")
    qh_data = pd.read_parquet(path_to_qh_data)
    qh_data = (
        qh_data.resample(
            rule="1min",
            closed="left",
            label="left",
        )
        .ffill()
        .to_parquet(path_to_save_resampled_data)
    )
    logger.info("Resampling of qh data done.")


if __name__ == "__main__":
    typer.run(main)
