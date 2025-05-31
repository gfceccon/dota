import polars as pl
from files import picks_bans_file
def get_picks_bans_data(path: str) -> pl.LazyFrame:
    """
    Carrega dados de picks e bans.
    
    Args:
        path (str): Caminho para os dados
        
    Returns:
        pl.LazyFrame: DataFrame com picks e bans
    """
    picks_bans = (
        pl.scan_csv(f"{path}/{picks_bans_file}")
        .select([
            "match_id",
            pl.col("is_pick").alias("pick_is_pick"),
            pl.col("hero_id").alias("pick_hero_id"), 
            pl.col("team").alias("pick_team"),
            pl.col("order").alias("pick_order"),
        ])
    )
    
    return picks_bans
