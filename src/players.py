import kagglehub
import polars as pl
from files import (
    players_file,
    picks_bans_file,
)


def get_players_draft(path: str, matches: pl.LazyFrame) -> pl.LazyFrame:
    """    Load players and draft data from CSV files and return a Polars LazyFrame.
    Args:
        path (str): The directory path where the CSV files are located.
    Returns:
        pl.LazyFrame: A Polars LazyFrame containing the players and draft data.
    """

    players_cols = [
        "kills", "deaths", "assists",
        "hero_id", "player_slot", "account_id",
        "obs_placed", "sen_placed",
        "gold_per_min", "xp_per_min",
        "hero_damage", "tower_damage", "hero_healing",
        "roshan_kills", "tower_kills",
    ]

    picks_cols = [
        "is_pick",
        "team",
        "hero_id",
    ]

    players = (
        pl.scan_csv(f"{path}/{players_file}")
        .select([pl.col(col).alias(f"player_{col}") for col in players_cols] + ["match_id"])
    )

    picks = (
        pl.scan_csv(f"{path}/{picks_bans_file}")
        .drop_nulls(subset="team")
        .select([pl.col(col).alias(f"pick_{col}") for col in picks_cols] + ["match_id"])
    )

    games = (
        matches
        .join(other=picks, on="match_id", how="inner")
        .join(other=players, left_on=["match_id", "pick_hero_id"], right_on=["match_id", "player_hero_id"], how="left")
        .with_columns([
            pl.when(pl.col("pick_is_pick") == False)
              .then(pl.col("pick_hero_id"))
              .otherwise(None)
              .alias("ban_hero_id"),
            pl.when(pl.col("pick_is_pick").eq(True)
                    & pl.col("pick_team").eq(0))
              .then(pl.col("pick_hero_id"))
              .otherwise(None)
              .alias("radiant_hero_id"),
            pl.when(pl.col("pick_is_pick") .eq(True)
                    & pl.col("pick_team").eq(1))
              .then(pl.col("pick_hero_id"))
              .otherwise(None)
              .alias("dire_hero_id"),
        ])
        .drop(["pick_hero_id", "pick_team", "pick_is_pick"])
        .select([
            "match_id",
            "radiant_hero_id", "dire_hero_id", "ban_hero_id",
            "player_kills", "player_deaths", "player_assists",
            pl.col("player_gold_per_min").alias("player_gpm"),
            pl.col("player_xp_per_min").alias("player_xpm"),
        ])
    )

    return games
