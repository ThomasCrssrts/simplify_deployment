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
    path_to_minute_data: Path = typer.Option(default=...),
    path_to_save_target: Path = typer.Option(default=...),
    number_of_weeks_to_use: int = 16,
) -> None:
    number_of_minutes_per_week = 7 * 24 * 60
    number_of_minutes_to_use = (
        number_of_minutes_per_week * number_of_weeks_to_use
    )

    logger.info("Reading in target data.")
    # Read in data
    minute_data: pd.DataFrame = pd.read_parquet(
        path_to_minute_data,
    )
    # Only use right amount of minutes
    logger.info(f"Filtering latest {number_of_weeks_to_use} weeks.")
    minute_data = minute_data.iloc[-number_of_minutes_to_use:, :]
    # Now select only right column
    target: pd.Series = minute_data.loc[:, "siCumulative"]
    # Minute 14 is our target
    target = target.loc[target.index.minute % 15 == 14]

    # Write to file
    logger.info("Writing target to file.")
    (target.to_frame().to_parquet(path_to_save_target))


if __name__ == "__main__":
    typer.run(main)
