import kagglehub
import polars as pl
from files import (
    players_file,
    draft_file,
)

def get_players_draft(path: str):
    """    Load players and draft data from CSV files and return a Polars LazyFrame.
    Args:
        path (str): The directory path where the CSV files are located.
    Returns:
        pl.LazyFrame: A Polars LazyFrame containing the players and draft data.
    """

    players_cols = [
        "match_id",
        "kills", "deaths", "assists",
        "obs_placed", "sen_placed",
        "gold_per_min", "xp_per_min",
        "hero_damage", "tower_damage", "hero_healing",
        "last_hits", "denies",
        "roshan_kills", "tower_kills",
    ]

    draft_cols = [
        "match_id", 
        "pick", 
        "active_team", 
        "player_slot",
        "hero_id",
    ]

    active_teams = [2, 3]

    players = (
        pl.scan_csv(f"{path}/{players_file}")
        .select([pl.col(col).alias(f"player_{col}") for col in players_cols if col != "match_id"] + ["match_id"])
    )

    draft = (
        pl.scan_csv(f"{path}/{draft_file}")
        .filter(pl.col("active_team").is_in(active_teams))
        .select([pl.col(col).alias(f"draft_{col}") for col in draft_cols if col != "match_id"] + ["match_id"])
    )

    game_players_pick = (
        players
        .drop_nulls(subset="draft_active_team")
        .filter(pl.col("draft_pick") == True)
        .filter(pl.col("draft_player_slot").str.len_chars() > 0)
        .join(other=draft, on="match_id", how="inner")
    )
    game_players_bans = (
        players
        .drop_nulls(subset="draft_active_team")
        .filter(pl.col("draft_pick") == False)
        .join(other=draft, on="match_id", how="inner")
    )

    return game_players_pick, game_players_bans
