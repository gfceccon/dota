import kagglehub
import polars as pl
from files import (
    players_file,
    draft_file,
)


active_teams = {
    "Radiant": 2,
    "Dire": 3,
}

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
        # "obs_placed", "sen_placed",
        "gold_per_min", "xp_per_min",
        # "hero_damage", "tower_damage", "hero_healing",
        # "last_hits", "denies",
        # "roshan_kills", "tower_kills",
    ]

    draft_cols = [
        "pick", 
        "active_team", 
        "player_slot",
        "hero_id",
    ]

    players = (
        pl.scan_csv(f"{path}/{players_file}")
        .select([pl.col(col).alias(f"player_{col}") for col in players_cols] + ["match_id"])
    )

    draft = (
        pl.scan_csv(f"{path}/{draft_file}")
        .drop_nulls(subset="active_team")
        .filter(pl.col("active_team").is_in(active_teams.values()))
        .select([pl.col(col).alias(f"draft_{col}") for col in draft_cols] + ["match_id"])
    )

    # Primeiro fazer o join e depois filtrar
    game_radiant_pick = (
        players
        .join(other=draft, on="match_id", how="inner")
        .filter(pl.col("draft_pick") == True)
        .filter(pl.col("draft_player_slot").str.len_chars() > 0)
        .filter(pl.col("draft_active_team") == active_teams["Radiant"])
    )

    game_dire_pick = (
        players
        .join(other=draft, on="match_id", how="inner")
        .drop_nulls(subset="draft_active_team")
        .filter(pl.col("draft_pick") == True)
        .filter(pl.col("draft_player_slot").str.len_chars() > 0)
        .filter(pl.col("draft_active_team") == active_teams["Dire"])
    )

    game_players_bans = (
        players
        .join(other=draft, on="match_id", how="inner")
        .drop_nulls(subset="draft_active_team")
        .filter(pl.col("draft_pick") == False)
        .join(other=draft, on="match_id", how="inner")
    )

    return game_radiant_pick, game_dire_pick, game_players_bans