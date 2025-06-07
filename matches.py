import kagglehub
import polars as pl
from files import (
    Dota2Files,
    get_lf,
)


def get_matches(path: str, patches: list[int], tier: list[str], min_duration=10 * 60, max_duration=120*60) -> pl.LazyFrame:
    leagues = (
        get_lf(Dota2Files.LEAGUES, path)
        .filter(pl.col("tier").is_in(tier))
        .select([
            pl.col("leagueid").alias("league_id"),
            pl.col("leaguename").alias("league_name"),
            pl.col("tier").alias("league_tier")
        ])
    )

    matches = (
        get_lf(Dota2Files.METADATA, path)
        .drop_nans(subset="match_id")
        .filter(
            pl.col("patch").is_in(patches),
            pl.col("duration").is_between(min_duration, max_duration),
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
