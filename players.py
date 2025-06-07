import kagglehub
import polars as pl
from files import (
    players_file,
    picks_bans_file,
    metadata_file
)
from heroes import get_heroes


def get_players_draft(path: str, matches: pl.LazyFrame) -> tuple[pl.LazyFrame, list[str], list[str]]:

    _players_cols = [
        "kills", "deaths", "assists",
        "hero_id", "player_slot", "account_id",
        "obs_placed", "sen_placed",
        "gold_per_min", "xp_per_min",
        "hero_damage", "tower_damage", "hero_healing",
        "roshan_kills", "tower_kills",
        "towers_killed", "roshans_killed",
        "last_hits", "denies",
        "aghanims_scepter", "aghanims_shard",
        "total_gold", "total_xp",
        "purchase_gem", "purchase_rapier",
    ]

    _picks_cols = [
        "is_pick",
        "team",
        "hero_id",
        "order",
    ]
    
    picks_cols = ["pick", "team", "hero_id", "hero_name", "order"]
    
    players_cols = [f"player_{col}" for col in _players_cols]

    players = (
        pl.scan_csv(f"{path}/{players_file}")
        .select([pl.col(col).alias(f"player_{col}") for col in _players_cols] + ["match_id"])
        .with_columns(pl.col("player_hero_id").alias("hero_id"))
    )

    picks = (
        pl.scan_csv(f"{path}/{picks_bans_file}")
        .drop_nulls(subset="team")
        .select([pl.col(col).alias(f"pick_{col}") for col in _picks_cols] + ["match_id"])
    )
    heroes, hero_cols, _, _ = get_heroes(path)
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
            (pl.col("hero_id") + 1).alias("hero_id"),

            *[
                pl.when(pl.col("pick_team").eq(team_id) &
                        pl.col("pick_is_pick").eq(True))
                .then(
                    pl.concat_list(
                        [pl.col(f"{stat}").is_null() for stat in players_cols]
                    ).list.any())
                .otherwise(pl.lit(None))
                .alias(f"{team_name}_stats_null")
                for team_id, team_name in [(0, "radiant"), (1, "dire")]
            ],
            *[
                pl.when(pl.col("pick_team").eq(team_id) &
                        pl.col("pick_is_pick").eq(True))
                .then(
                    pl.col("roles"))
                .otherwise(pl.lit(None))
                .alias(f"{team_name}_hero_roles")
                for team_id, team_name in [(0, "radiant"), (1, "dire")]
            ],
            
            *[pl.col(col).cast(pl.Float64, strict=False).fill_null(strategy="zero").alias(col) for col in players_cols],
        ])
        .select([
            "match_id",
            "radiant_hero_roles",
            "dire_hero_roles",
            "radiant_stats_null",
            "dire_stats_null",
            *picks_cols,
            *hero_cols,
            *players_cols,
        ])
    )

    return games, players_cols, hero_cols


if __name__ == "__main__":
    dataset_name = "bwandowando/dota-2-pro-league-matches-2023"
    path = kagglehub.dataset_download(dataset_name)
    matches = pl.scan_csv(f"{path}/{metadata_file}").select(["match_id"])
    players, player_cols, hero_cols = get_players_draft(path, matches)

    print(f"Players DataFrame: {players.collect().head()}")
