import kagglehub
import polars as pl
from src.files import (
    players_file,
    picks_bans_file,
)


def get_players_draft(path: str, matches: pl.LazyFrame) -> pl.LazyFrame:

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
        .join(picks, on="match_id", how="inner")
        .join(players, left_on=["match_id", "pick_hero_id"], right_on=["match_id", "player_hero_id"], how="left")
        .with_columns([
            # Hero assignments based on pick/ban and team
            pl.when(~pl.col("pick_is_pick"))
              .then(pl.col("pick_hero_id"))
              .alias("ban_hero_id"),
            
            pl.when(pl.col("pick_is_pick") & (pl.col("pick_team") == 0))
              .then(pl.col("pick_hero_id"))
              .alias("radiant_hero_id"),
            
            pl.when(pl.col("pick_is_pick") & (pl.col("pick_team") == 1))
              .then(pl.col("pick_hero_id"))
              .alias("dire_hero_id"),
            
            # Team-based stats assignment using dictionary comprehension
            *[
                pl.when(pl.col("pick_team") == team_id)
                  .then(pl.col(f"player_{stat}"))
                  .alias(f"{team_name}_{stat}")
                for team_id, team_name in [(0, "radiant"), (1, "dire")]
                for stat in ["gold_per_min", "xp_per_min", "kills", "deaths", "assists"]
            ]
        ])
        .select([
            "match_id",
            "radiant_hero_id", "dire_hero_id", "ban_hero_id",
            "radiant_kills", "dire_kills",
            "radiant_deaths", "dire_deaths", 
            "radiant_assists", "dire_assists",
            "radiant_gold_per_min", "dire_gold_per_min",
            "radiant_xp_per_min", "dire_xp_per_min",
        ])
    )

    return games
