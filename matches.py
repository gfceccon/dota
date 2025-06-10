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
    match_cols = [
        "duration_normalized",
        "winner"
    ]
    matches = (
        get_lf(Dota2Files.METADATA, path, (2020, 2025))
        .drop_nans(subset="match_id")
        .filter(
            pl.col("patch").is_in(patches),
            pl.col("duration").is_between(min_duration, max_duration),
            pl.col("radiant_win").is_not_null(),
        )
        .with_columns(
            (pl.col("duration") / max_duration).alias("duration_normalized").cast(pl.Float32),
            pl.col("radiant_win").cast(pl.Int32).alias("winner"),
            pl.col("leagueid").alias("league_id"),
        )
        .select([
            "match_id",
            "league_id",
            *match_cols,
        ])
        .join(other=leagues, on="league_id", how="left")
    )

    return matches
