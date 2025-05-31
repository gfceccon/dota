import polars as pl
from files import heroes_file, picks_bans_file
from typing import Dict


def get_hero_mapping(path: str) -> Dict[int, int]:
    """
    Carrega o mapeamento de hero_id para índices sequenciais (0-126).
    
    Args:
        path (str): Caminho para os dados
        
    Returns:
        Dict[int, int]: Mapeamento de hero_id para índice
    """
    heroes_df = pl.read_csv(f"{path}/{heroes_file}")
    hero_mapping = {}
    
    for i, row in enumerate(heroes_df.iter_rows(named=True)):
        hero_mapping[row['id']] = i
    
    return hero_mapping

