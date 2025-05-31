import polars as pl
features_example = {
    # Game duration
    "game_duration": 1000,
    
    # The maximum number of heroes is defined by the game, e.g., Dota 2 has approximately 120 heroes.

    # Heroes (radiant)
    # Each radiant hero is represented by a binary value indicating whether it was picked or not.
    # The keys are named "radiant_hero_0", "radiant_hero_1", ..., "radiant_hero_n" for the n-th hero.
    "radiant_hero_0": 1,
    "radiant_hero_1": 0,
    "radiant_hero_3": 1,
    #... maximum n heroes

    # Heroes (banned)
    # Each hero is represented by a binary value indicating whether it was banned or not.
    # The keys are named "banned_hero_0", "banned_hero_1", ..., "banned_hero_n" for the n-th hero.
    "banned_hero_0": 1,
    "banned_hero_1": 0,
    "banned_hero_2": 1,
    #... maximum n heroes

    # Heroes (picked by dire)
    # Each dire hero is represented by a binary value indicating whether it was picked by the opponent or not.
    # The keys are named "dire_hero_0", "dire_hero_1", ..., "dire_hero_n" for the n-th hero.
    "dire_hero_0": 0,
    "dire_hero_1": 1,
    "dire_hero_2": 0,
    #... maximum n heroes

    # Objectives
    # Number of kills, deaths, and assists for each hero.
    # The keys are named "kills_0", "kills_1", ..., "kills_n" for the n-th hero.
    # The same applies for deaths and assists.
    "kills_hero_0": 2,
    "kills_hero_1": 1,
    "kills_hero_2": 0,
    # ... maximum n heroes

    "deaths_hero_0": 2,
    "deaths_hero_1": 1,
    "deaths_hero_2": 0,
    # ... maximum n heroes

    "assists_hero_0": 2,
    "assists_hero_1": 1,
    "assists_hero_2": 0,
    # ... maximum n heroes

    # Gold per minute (GPM)
    # The keys are named "gold_0", "gold_1", ..., "gold_n" for the n-th hero.
    "gpm_hero_0": 0,
    "gpm_hero_1": 200,
    "gpm_hero_2": 0,
    
    # Experience per minute (XPM)
    # The keys are named "xpm_0", "xpm_1", ..., "xpm_n" for the n-th hero.
    "xpm_hero_0": 0,
    "xpm_hero_1": 0,
    "xpm_hero_2": 0,
}

def get_feature_dict(n_heroes: int) -> dict:
    """
    Cria um schema de features para o dataset de Dota 2.

    Args:
        n_heroes (int): Número total de heróis no jogo.

    Returns:
        pl.Schema: Schema com as features definidas.
    """
    schema = {}

    # Adiciona as features de heróis
    for i in range(n_heroes):
        schema[f"radiant_hero_{i}"] = pl.Boolean
        schema[f"banned_hero_{i}"] = pl.Boolean
        schema[f"dire_hero_{i}"] = pl.Boolean
        schema[f"kills_hero_{i}"] = pl.Int64
        schema[f"deaths_hero_{i}"] = pl.Int64
        schema[f"assists_hero_{i}"] = pl.Int64
        schema[f"gpm_hero_{i}"] = pl.Int64
        schema[f"xpm_hero_{i}"] = pl.Int64
    
    schema["game_duration"] = pl.Int64,

    return schema