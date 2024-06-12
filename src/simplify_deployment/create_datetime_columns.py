import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import typer
from dateutil import tz
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

# Create logger
logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)

# Create handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Add handler to logger
logger.addHandler(handler)


def periodic_spline_transformer(
    period, n_splines=None, degree=3, extrapolation: str = "periodic"
):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation=extrapolation,
        include_bias=True,
    ).set_output(transform="pandas")


def create_datetime_columns(
    X: pd.Series,
    kind: Literal["ordinal", "onehot", "spline"] = "ordinal",
    timezone: str = "Europe/Brussels",
) -> pd.DataFrame:
    local_time_index = X.index.tz_convert(tz.gettz(timezone))
    X_datetime = pd.DataFrame(index=X.index)
    X_datetime["year"] = local_time_index.year
    X_datetime["month"] = local_time_index.month
    X_datetime["weekday"] = local_time_index.weekday
    X_datetime["hour"] = local_time_index.hour
    X_datetime["minute"] = local_time_index.minute
    match kind:
        case "ordinal":
            return X_datetime
        case "onehot":
            encoder = OneHotEncoder(
                categories=[
                    range(2019, 2025),
                    range(1, 13),
                    range(0, 7),
                    range(0, 24),
                    range(0, 60),
                ],
                sparse_output=False,
            ).set_output(transform="pandas")
            X_datetime_one_hot = encoder.fit_transform(X_datetime)
            return X_datetime_one_hot
        case "spline":
            month_splines = periodic_spline_transformer(12).fit_transform(
                X_datetime[["month"]]
            )
            day_splines = periodic_spline_transformer(7).fit_transform(
                X_datetime[["weekday"]]
            )
            hour_splines = periodic_spline_transformer(24).fit_transform(
                X_datetime[["hour"]]
            )
            minute_splines = periodic_spline_transformer(60).fit_transform(
                X_datetime[["minute"]]
            )
            return pd.concat(
                [
                    month_splines,
                    day_splines,
                    hour_splines,
                    minute_splines,
                    X_datetime["year"],
                ],
                axis=1,
            )


def main(
    path_to_X: Path = typer.Option(default=...),
    kind: Literal["ordinal", "onehot", "spline"] = typer.Option(default=...),
    path_output: Path = typer.Option(default=...),
):
    logger.info("Reading in data for datetime transformation.")
    X = pd.read_parquet(path_to_X)
    X_datetime = create_datetime_columns(
        X,
        kind,
    )
    X_datetime.to_parquet(path_output)
    logger.info("Done making datetime columns.")


if __name__ == "__main__":
    typer.run(main)
