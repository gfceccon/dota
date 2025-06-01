import polars as pl
from matches import get_matches
from players import get_players_draft


def preprocess_dota_dataset(path: str, patches: list[int], tier: list[str],
                            min_duration: int = 10 * 60) -> pl.DataFrame:
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

    # Pré-processar dados
    print("Iniciando pré-processamento...")

    # Carregando e filtrando partidas
    matches = get_matches(path, patches, tier, min_duration)

    # Carregando jogos e draft de jogadores
    games = get_players_draft(path, matches)

    # Agrupando dados por partida
    dataset = games.group_by("match_id").agg(pl.all())
    dataset_collected = dataset.collect()

    print("Pré-processamento concluído!")
    return dataset_collected