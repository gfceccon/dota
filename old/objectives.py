import polars as pl
from files import (
    Dota2Files,
    get_lf,
)

def get_objectives(path: str) -> pl.LazyFrame:

    cols = [
        "match_id", "time", "type", "value", "killer", "team",
        "slot", "key", "player_slot", "unit"
    ]

    obj_types = {
        "CHAT_MESSAGE_AEGIS": "aegis",
        "CHAT_MESSAGE_AEGIS_STOLEN": "aegis_stolen",
        "CHAT_MESSAGE_COURIER_LOST": "courier_lost",
        "CHAT_MESSAGE_DENIED_AEGIS": "denied_aegis",
        "CHAT_MESSAGE_FIRSTBLOOD": "first_blood",
        "CHAT_MESSAGE_ROSHAN_KILL": "roshan_kill",
        "building_kill": "building_kill",
    }

    objectives = (
        get_lf(Dota2Files.OBJECTIVES, path)
        .with_columns([pl.col(f"obj_{col}") for col in cols if col != "match_id" and col != "type"] + [
            pl.col("type")
            .replace(obj_types)
            .alias("obj_type"),
            pl.col("match_id")
        ])
    )

    return objectives
