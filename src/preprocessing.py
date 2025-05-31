import polars as pl
import numpy as np
from typing import Dict, List, Tuple
from heroes import get_hero_mapping
from matches import get_matches
from objectives import get_objectives
from files import heroes_file, picks_bans_file, players_file
from picks import get_picks_bans_data
from players import get_players_draft
from features import get_feature_dict


def preprocess_dota_dataset(path: str, patches: List[int], tier: List[str],
                            min_duration: int = 10 * 60) -> pl.LazyFrame:
    """
    Função principal para pré-processamento do dataset de Dota 2.

    Args:
        path (str): Caminho para os dados
        patches (List[int]): Lista de patches para filtrar
        tier (List[str]): Lista de tiers de liga para filtrar
        min_duration (int): Duração mínima do jogo em segundos

    Returns:
        pl.LazyFrame: Dataset pré-processado com todas as features
    """

    # Carregar dados básicos
    print("Carregando dados básicos...")
    matches = get_matches(path, patches, tier, min_duration)
    hero_mapping = get_hero_mapping(path)
    num_heroes = len(hero_mapping)
    print(f"Encontrados {num_heroes} heróis")

    # Carregar dados de picks/bans
    print("Carregando dados de picks e bans...")
    picks_bans = get_picks_bans_data(path)

    # Carregar dados de picks e bans dos jogadores, incluindo stats
    print("Carregando dados de players...")
    game_radiant_pick, game_dire_pick, game_players_bans = get_players_draft(
        path)

    # Processar picks e bans para criar features binárias de heróis
    print("Processando picks e bans...")
    
    # Separar picks por time (Radiant = 0, Dire = 1)
    radiant_picks = (
        picks_bans
        .filter(pl.col("pick_is_pick") == True)
        .filter(pl.col("pick_team") == 0)  # Radiant
        .with_columns([
            pl.col("pick_hero_id").replace(hero_mapping).alias("hero_idx")
        ])
        .group_by("match_id")
        .agg([
            pl.col("hero_idx").alias("radiant_heroes")
        ])
    )
    
    dire_picks = (
        picks_bans
        .filter(pl.col("pick_is_pick") == True)
        .filter(pl.col("pick_team") == 1)  # Dire
        .with_columns([
            pl.col("pick_hero_id").replace(hero_mapping).alias("hero_idx")
        ])
        .group_by("match_id")
        .agg([
            pl.col("hero_idx").alias("dire_heroes")
        ])
    )
    
    # Processar bans (todos os heróis banidos independente do time)
    bans = (
        picks_bans
        .filter(pl.col("pick_is_pick") == False)  # Bans
        .with_columns([
            pl.col("pick_hero_id").replace(hero_mapping).alias("hero_idx")
        ])
        .group_by("match_id")
        .agg([
            pl.col("hero_idx").alias("banned_heroes")
        ])
    )
    
    # Processar dados dos jogadores para stats
    print("Processando stats dos jogadores...")
    
    # Processar stats do Radiant por herói
    radiant_stats = (
        game_radiant_pick
        .with_columns([
            pl.col("draft_hero_id").replace(hero_mapping).alias("hero_idx")
        ])
        .group_by(["match_id", "hero_idx"])
        .agg([
            pl.col("player_kills").sum().alias("kills"),
            pl.col("player_deaths").sum().alias("deaths"), 
            pl.col("player_assists").sum().alias("assists"),
            pl.col("player_gold_per_min").mean().alias("gpm"),
            pl.col("player_xp_per_min").mean().alias("xpm")
        ])
    )
    
    # Processar stats do Dire por herói
    dire_stats = (
        game_dire_pick
        .with_columns([
            pl.col("draft_hero_id").replace(hero_mapping).alias("hero_idx")
        ])
        .group_by(["match_id", "hero_idx"])
        .agg([
            pl.col("player_kills").sum().alias("kills"),
            pl.col("player_deaths").sum().alias("deaths"),
            pl.col("player_assists").sum().alias("assists"), 
            pl.col("player_gold_per_min").mean().alias("gpm"),
            pl.col("player_xp_per_min").mean().alias("xpm")
        ])
    )
    
    # Criar features iniciais com zeros para todos os heróis
    print("Criando features binárias de heróis...")
    
    base_features = matches.with_columns([
        # Inicializar todas as features de heróis com valores padrão
        *[pl.lit(False).alias(f"radiant_hero_{i}") for i in range(num_heroes)],
        *[pl.lit(False).alias(f"dire_hero_{i}") for i in range(num_heroes)],
        *[pl.lit(False).alias(f"banned_hero_{i}") for i in range(num_heroes)],
        *[pl.lit(0).alias(f"kills_hero_{i}") for i in range(num_heroes)],
        *[pl.lit(0).alias(f"deaths_hero_{i}") for i in range(num_heroes)],
        *[pl.lit(0).alias(f"assists_hero_{i}") for i in range(num_heroes)],
        *[pl.lit(0.0).alias(f"gpm_hero_{i}") for i in range(num_heroes)],
        *[pl.lit(0.0).alias(f"xpm_hero_{i}") for i in range(num_heroes)]
    ])
    
    # Aplicar features de picks do Radiant
    print("Aplicando features de picks...")
    for i in range(num_heroes):
        radiant_pick_i = (
            radiant_picks
            .filter(pl.col("radiant_heroes").list.contains(i))
            .select("match_id")
            .with_columns(pl.lit(True).alias(f"radiant_hero_{i}"))
        )
        base_features = base_features.join(radiant_pick_i, on="match_id", how="left", suffix="_new")
        base_features = base_features.with_columns([
            pl.coalesce([pl.col(f"radiant_hero_{i}_new"), pl.col(f"radiant_hero_{i}")]).alias(f"radiant_hero_{i}")
        ]).drop(f"radiant_hero_{i}_new")
    
    # Aplicar features de picks do Dire
    for i in range(num_heroes):
        dire_pick_i = (
            dire_picks
            .filter(pl.col("dire_heroes").list.contains(i))
            .select("match_id")
            .with_columns(pl.lit(True).alias(f"dire_hero_{i}"))
        )
        base_features = base_features.join(dire_pick_i, on="match_id", how="left", suffix="_new")
        base_features = base_features.with_columns([
            pl.coalesce([pl.col(f"dire_hero_{i}_new"), pl.col(f"dire_hero_{i}")]).alias(f"dire_hero_{i}")
        ]).drop(f"dire_hero_{i}_new")
    
    # Aplicar features de bans
    for i in range(num_heroes):
        ban_i = (
            bans
            .filter(pl.col("banned_heroes").list.contains(i))
            .select("match_id")
            .with_columns(pl.lit(True).alias(f"banned_hero_{i}"))
        )
        base_features = base_features.join(ban_i, on="match_id", how="left", suffix="_new")
        base_features = base_features.with_columns([
            pl.coalesce([pl.col(f"banned_hero_{i}_new"), pl.col(f"banned_hero_{i}")]).alias(f"banned_hero_{i}")
        ]).drop(f"banned_hero_{i}_new")
    
    # Aplicar stats dos jogadores
    print("Aplicando stats dos jogadores...")
    for i in range(num_heroes):
        # Stats do Radiant
        radiant_stats_i = (
            radiant_stats
            .filter(pl.col("hero_idx") == i)
            .select([
                "match_id", "kills", "deaths", "assists", "gpm", "xpm"
            ])
            .rename({
                "kills": f"kills_hero_{i}",
                "deaths": f"deaths_hero_{i}",
                "assists": f"assists_hero_{i}",
                "gpm": f"gpm_hero_{i}",
                "xpm": f"xpm_hero_{i}"
            })
        )
        
        base_features = base_features.join(radiant_stats_i, on="match_id", how="left", suffix="_new")
        base_features = base_features.with_columns([
            pl.coalesce([pl.col(f"kills_hero_{i}_new"), pl.col(f"kills_hero_{i}")]).alias(f"kills_hero_{i}"),
            pl.coalesce([pl.col(f"deaths_hero_{i}_new"), pl.col(f"deaths_hero_{i}")]).alias(f"deaths_hero_{i}"),
            pl.coalesce([pl.col(f"assists_hero_{i}_new"), pl.col(f"assists_hero_{i}")]).alias(f"assists_hero_{i}"),
            pl.coalesce([pl.col(f"gpm_hero_{i}_new"), pl.col(f"gpm_hero_{i}")]).alias(f"gpm_hero_{i}"),
            pl.coalesce([pl.col(f"xpm_hero_{i}_new"), pl.col(f"xpm_hero_{i}")]).alias(f"xpm_hero_{i}")
        ]).drop([f"kills_hero_{i}_new", f"deaths_hero_{i}_new", f"assists_hero_{i}_new", f"gpm_hero_{i}_new", f"xpm_hero_{i}_new"])
        
        # Stats do Dire
        dire_stats_i = (
            dire_stats
            .filter(pl.col("hero_idx") == i)
            .select([
                "match_id", "kills", "deaths", "assists", "gpm", "xpm"
            ])
            .rename({
                "kills": f"kills_hero_{i}",
                "deaths": f"deaths_hero_{i}",
                "assists": f"assists_hero_{i}",
                "gpm": f"gpm_hero_{i}",
                "xpm": f"xpm_hero_{i}"
            })
        )
        
        base_features = base_features.join(dire_stats_i, on="match_id", how="left", suffix="_new")
        base_features = base_features.with_columns([
            pl.coalesce([pl.col(f"kills_hero_{i}_new"), pl.col(f"kills_hero_{i}")]).alias(f"kills_hero_{i}"),
            pl.coalesce([pl.col(f"deaths_hero_{i}_new"), pl.col(f"deaths_hero_{i}")]).alias(f"deaths_hero_{i}"),
            pl.coalesce([pl.col(f"assists_hero_{i}_new"), pl.col(f"assists_hero_{i}")]).alias(f"assists_hero_{i}"),
            pl.coalesce([pl.col(f"gpm_hero_{i}_new"), pl.col(f"gpm_hero_{i}")]).alias(f"gpm_hero_{i}"),
            pl.coalesce([pl.col(f"xpm_hero_{i}_new"), pl.col(f"xpm_hero_{i}")]).alias(f"xpm_hero_{i}")
        ]).drop([f"kills_hero_{i}_new", f"deaths_hero_{i}_new", f"assists_hero_{i}_new", f"gpm_hero_{i}_new", f"xpm_hero_{i}_new"])
    
    final_dataset = base_features


    print("Pré-processamento concluído!")
    return final_dataset


def save_preprocessed_dataset(dataset: pl.LazyFrame, output_path: str):
    """
    Salva o dataset pré-processado.

    Args:
        dataset: Dataset pré-processado
        output_path: Caminho para salvar o arquivo
    """
    print(f"Salvando dataset em {output_path}...")
    dataset.collect().write_csv(output_path)
    print("Dataset salvo com sucesso!")


if __name__ == "__main__":
    # Exemplo de uso
    import kagglehub

    dataset_name = "bwandowando/dota-2-pro-league-matches-2023/versions/177"
    path = kagglehub.dataset_download(dataset_name)

    # Pré-processar dados
    dataset = preprocess_dota_dataset(
        path=path,
        patches=[54],  # Patch 7.34
        tier=["professional"],
        min_duration=10 * 60  # 10 minutos
    )

    print(
    )
    
    # Salvar dataset pré-processado
    output_path = "preprocessed_dota_dataset.csv"
    save_preprocessed_dataset(dataset, output_path)
