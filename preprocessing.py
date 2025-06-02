import polars as pl
from matches import get_matches
from players import get_players_draft


def preprocess_dota_dataset(path: str, patches: list[int], tier: list[str],
                            min_duration: int = 10 * 60, max_duration: int = 120 * 60) -> pl.DataFrame:
    # Pré-processar dados
    print("Iniciando pré-processamento...")

    # Carregando e filtrando partidas
    matches = get_matches(path, patches, tier, min_duration, max_duration)

    # Carregando jogos e draft de jogadores
    games = get_players_draft(path, matches)

    # Agrupando dados por partida
    dataset = games.group_by("match_id", maintain_order=True).agg(pl.all().drop_nulls()) 
    dataset_collected = dataset.collect()

    print("Pré-processamento concluído!")
    return dataset_collected
