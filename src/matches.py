import kagglehub
import polars as pl
from files import (
    metadata_file,
    leagues_file
)


def get_matches(path: str, patches: list[int], tier: list[str], min_duration=10 * 60) -> pl.LazyFrame:
    """
    Load matches data from CSV files and return a Polars LazyFrame.

    Args:
        path (str): The directory path where the CSV files are located.
        patches (list[int]): List of patch versions to filter matches.
        tier (list[str]): List of league tiers to filter matches.
        min_duration (int): Minimum game duration in seconds to filter matches. Default is 10 minutes (600 seconds).

    Returns:
        pl.LazyFrame: A Polars LazyFrame containing the matches data.
        Columns include:
            - match_id
            - league_id
            - game_duration
    """

    leagues = (
        pl.scan_csv(f"{path}/{leagues_file}")
        .filter(pl.col("tier").is_in(tier))
        .select([
            pl.col("leagueid").alias("league_id"),
            pl.col("leaguename").alias("league_name"),
            pl.col("tier").alias("league_tier")
        ])
    )

    matches = (
        pl.scan_csv(f"{path}/{metadata_file}")
        .drop_nans(subset="match_id")
        .filter(
            pl.col("patch").is_in(patches),
            pl.col("duration") >= min_duration,
        )
        .select([
            "match_id",
            pl.col("leagueid").alias("league_id"),
            pl.col("duration").alias("game_duration"),
        ])
        .join(other=leagues, on="league_id", how="left")
        .drop("league_id")
    )

    return matches
