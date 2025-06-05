import kagglehub
import polars as pl
from files import (
    players_file,
    picks_bans_file,
)
from heroes import get_heroes

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
    "order",
]

game_cols = ["player_gold_per_min", "player_xp_per_min",
             "player_kills", "player_deaths", "player_assists"]
hero_cols = ["hero_id", "hero_name", "hero_primary_attr",
             "hero_attack_type", "hero_roles"]


def get_players_draft(path: str, matches: pl.LazyFrame) -> pl.LazyFrame:

    players = (
        pl.scan_csv(f"{path}/{players_file}")
        .select([pl.col(col).alias(f"player_{col}") for col in players_cols] + ["match_id"])
    )

    picks = (
        pl.scan_csv(f"{path}/{picks_bans_file}")
        .drop_nulls(subset="team")
        .select([pl.col(col).alias(f"pick_{col}") for col in picks_cols] + ["match_id"])
    )

    heroes = get_heroes(path).select(
        pl.col("hero_id").alias("pick_hero_id"),
        pl.col("hero_name").alias("pick_hero_name"),
        pl.col("primary_attr").alias("pick_primary_attr"),
        pl.col("roles").alias("pick_roles")
    ).lazy()

    games = (
        matches
        .join(picks, on="match_id", how="inner")
        .join(players, left_on=["match_id", "pick_hero_id"], right_on=["match_id", "player_hero_id"], how="left")
        .join(heroes, left_on="pick_hero_id", right_on="hero_id", how="inner")
        .with_columns([
            # Renomeia as colunas para facilitar o acesso
            pl.col("pick_team").alias("team"),
            pl.col("pick_hero_id").alias("hero"),
            pl.col("pick_is_pick").alias("pick"),
            pl.col("pick_order").alias("order"),
            pl.col("hero_id"),


            # Team-based stats assignment using dictionary comprehension
            *[
                pl.when(pl.col("pick_team") == team_id,
                        pl.col("pick_is_pick") == True)
                .then(
                    pl.concat_list(
                        [f"{stat}" for stat in game_cols]
                    ).alias(f"player_stats"))
                .otherwise(pl.lit(None))
                .alias(f"{team_name}_stats")
                for team_id, team_name in [(0, "radiant"), (1, "dire")]
            ],
            *[
                pl.when(pl.col("pick_team") == team_id,
                        pl.col("pick_is_pick") == True)
                .then(
                    pl.concat_list(
                        [f"{stat}" for stat in hero_cols]
                    ).alias(f"hero_stats"))
                .otherwise(pl.lit(None))
                .alias(f"{team_name}_hero_stats")
                for team_id, team_name in [(0, "radiant"), (1, "dire")]
            ],
        ])
        .select([
            "match_id",
            "hero_id", "hero_name"
            "team", "order",
            "hero", "pick",
            "radiant_stats",
            "radiant_hero_stats",
            "dire_stats",
            "dire_hero_stats",
            "pick_order",
            *game_cols
        ])
    )

    return games
