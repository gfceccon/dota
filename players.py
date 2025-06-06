import kagglehub
import polars as pl
from files import (
    players_file,
    picks_bans_file,
)
from heroes import get_heroes


def get_players_draft(path: str, matches: pl.LazyFrame) -> tuple[pl.LazyFrame, list[str], list[str]]:

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
    hero_cols = ["primary_attribute", "attack_type", "roles", ]

    players = (
        pl.scan_csv(f"{path}/{players_file}")
        .select([pl.col(col).alias(f"player_{col}") for col in players_cols] + ["match_id"])
        .with_columns(pl.col("player_hero_id").alias("hero_id"))
    )

    picks = (
        pl.scan_csv(f"{path}/{picks_bans_file}")
        .drop_nulls(subset="team")
        .select([pl.col(col).alias(f"pick_{col}") for col in picks_cols] + ["match_id"])
    )
    heroes, _, _ = get_heroes(path)
    games = (
        matches
        .join(picks, on="match_id", how="inner")
        .with_columns(pl.col("pick_hero_id").alias("hero_id"))
        .join(players, left_on=["match_id", "hero_id"], right_on=["match_id", "hero_id"], how="left")
        .join(heroes, on="hero_id", how="inner")
        .with_columns([
            # Renomeia as colunas para facilitar o acesso
            pl.col("pick_team").alias("team"),
            pl.col("pick_is_pick").cast(pl.Int32).alias("pick"),
            pl.col("pick_order").alias("order"),


            # Team-based stats assignment using dictionary comprehension
            *[
                pl.when(pl.col("pick_team") == team_id,
                        pl.col("pick_is_pick") == True)
                .then(
                    pl.concat_list(
                        [pl.col(f"{stat}") * 1.0 for stat in game_cols]
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
                .alias(f"{team_name}_hero_roles")
                for team_id, team_name in [(0, "radiant"), (1, "dire")]
            ],
        ]).select([
            "match_id",
            "hero_id",
            "hero_name",
            "team",
            "order",
            "pick",
            "radiant_stats",
            "radiant_hero_roles",
            "dire_stats",
            "dire_hero_roles",
            *game_cols
        ])
    )

    return games, game_cols, hero_cols
