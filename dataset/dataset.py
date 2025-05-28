import kagglehub
import polars as pl
import ast
from files import (
    metadata_file,
    objectives_file,
    picks_bans_file,
    players_file,
    leagues_file
)


dataset_name = "bwandowando/dota-2-pro-league-matches-2023/versions/177"
path = kagglehub.dataset_download(dataset_name)

columns = {
    "metadata": ["match_id", "duration", "patch", "leagueid"],
    "objectives": ["time", "type", "value", "killer", "team", "slot", "key", "player_slot", "unit", "match_id"],
    "picks_bans": ["is_pick", "hero_id", "team", "order", "match_id"],
    "players": ["account_id", "kills", "deaths", "assists",
                "gold_t", "lh_t", "dn_t", "xp_t",
                "gold_per_min", "xp_per_min", "net_worth", "level",
                "hero_damage", "tower_damage", "hero_healing",
                "last_hits", "denies",
                "roshan_kills", "tower_kills",
                "match_id",
                "leaver_status",
                ],
}

dataset_cols = [
    'match_id', 'game_duration', 'game_patch',
    'league_id', 'league_name', 'league_tier',
    'obj_time', 'obj_type', 'obj_value', 'obj_killer', 'obj_team', 'obj_slot', 'obj_key', 'obj_player_slot', 'obj_unit',
    'pick_is_pick', 'pick_hero_id', 'pick_team', 'pick_order', 'player_kills', 'player_deaths', 'player_assists',
    'player_gold_per_min', 'player_xp_per_min', 'player_net_worth',
    'player_level', 'player_hero_damage', 'player_tower_damage', 'player_hero_healing',
    'player_last_hits', 'player_denies', 'player_roshan_kills', 'player_tower_kills'
]

leagues = (
    pl.scan_csv(f"{path}/{leagues_file}")
    .filter(pl.col("tier") == "professional")
    .select([
        pl.col("leagueid").alias("league_id"),
        pl.col("leaguename").alias("league_name"),
        pl.col("tier").alias("league_tier")
    ])
)

metadata = (
    pl.scan_csv(f"{path}/{metadata_file}")
    .drop_nans(subset="match_id")
    .select([pl.col(col).alias(f"game_{col}") for col in columns["metadata"] if col != "leagueid" and col != "match_id"] +
            [pl.col("leagueid").alias("league_id"), "match_id"])
    .join(other=leagues, on="league_id", how="left")
)


objectives = (
    pl.scan_csv(f"{path}/{objectives_file}")
    .drop_nans(subset="match_id")
    .select([pl.col(col).alias(f"obj_{col}") for col in columns["objectives"] if col != "match_id"] + ["match_id"])
)

picks_bans = (
    pl.scan_csv(f"{path}/{picks_bans_file}")
    .drop_nans(subset="match_id")
    .select([pl.col(col).alias(f"pick_{col}") for col in columns["picks_bans"] if col != "match_id"] + ["match_id"])
)

players = (
    pl.scan_csv(f"{path}/{players_file}")
    .drop_nans(subset="match_id")
    .select([pl.col(col).alias(f"player_{col}") for col in columns["players"] if col != "match_id"] + ["match_id"])
)

game_objectives = (
    metadata.join(other=objectives, on="match_id",
                  how="left", suffix="_objectives")
    .group_by("match_id", maintain_order=True)
    .agg([pl.col(f"obj_{col}") for col in columns["objectives"] if col != "match_id"], )
)

game_picks_bans = (
    metadata.join(other=picks_bans, on="match_id",
                  how="left", suffix="_picks_bans")
    .group_by("match_id", maintain_order=True)
    .agg([pl.col(f"pick_{col}") for col in columns["picks_bans"] if col != "match_id"], )
)

game_players = (
    metadata.join(other=players, on="match_id", how="left", suffix="_players")
    .group_by("match_id", maintain_order=True)
    .agg([pl.col(f"player_{col}") for col in columns["players"] if col != "match_id"], )
)

dataset = (
    metadata
    .join(other=game_objectives, on="match_id", how="left")
    .join(other=game_picks_bans, on="match_id", how="left")
    .join(other=game_players, on="match_id", how="left")
)

print(
    players.select("player_leaver_status").unique().collect()
)
