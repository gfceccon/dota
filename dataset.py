import ast
import kagglehub
import polars as pl
from heroes import get_heroes
from patches import get_patches
from matches import get_matches
from players import get_players_draft, game_cols
from players import game_cols
import ast


def preprocess_dataset(path: str, patches: list[int], tier: list[str],
                       min_duration: int = 10 * 60, max_duration: int = 120 * 60) -> pl.LazyFrame:
    # Carregando e filtrando partidas
    matches = get_matches(path, patches, tier, min_duration, max_duration)

    # Carregando jogos e draft de jogadores
    games = get_players_draft(path, matches)

    # Agrupando dados por partida
    dataset = (
        games.group_by("match_id")
        .agg(
            pl.all().drop_nulls(),
            
            pl.when(pl.col("team") == 0, pl.col("pick")).then(
                pl.col("hero")).drop_nulls().alias("radiant_picks"),
            pl.when(pl.col("team") == 1, pl.col("pick")).then(
                pl.col("hero")).drop_nulls().alias("dire_picks"),
            
            pl.when(pl.col("team") == 0, ~pl.col("pick")).then(
                pl.col("hero")).drop_nulls().alias("radiant_bans"),
            pl.when(pl.col("team") == 1, ~pl.col("pick")).then(
                pl.col("hero")).drop_nulls().alias("dire_bans"),
            
            pl.concat_list([pl.col(f"{col}").max()
                           for col in game_cols]).alias("max_stats"),
        ))

    return dataset


def get_dataset(path: str, years: tuple[int, int] = (2023, 2024), tier: list[str] = ['professional'], duration: tuple[int, int] = (30, 120)) -> pl.DataFrame:
    patches = get_patches(path, begin_year=years[0], end_year=years[1])
    dataset = preprocess_dataset(
        path,
        list(patches.keys()),
        tier,
        duration[0] * 60,
        duration[1] * 60
    )
    return dataset.collect()


def save_dataset(dataset: pl.DataFrame, output_path: str = "./tmp/DATASET.json") -> None:
    print(f"Salvando dataset em {output_path}...")
    dataset.write_json(output_path)
    print("Dataset salvo com sucesso!")


if __name__ == "__main__":

    # Download do dataset
    dataset_name = "bwandowando/dota-2-pro-league-matches-2023/versions/177"
    path = kagglehub.dataset_download(dataset_name)
    print(f"Dataset baixado para: {path}")

    # Definir patches e tier
    patches = get_patches(path, begin_year=2023,
                          end_year=2024)  # Patches disponíveis
    tier = ["professional"]  # Tier de interesse

    print(f"Patches disponíveis: {patches}")
    sum = 0
    for patch, (count, _) in patches.items():
        sum += count
    print(f"- Total de jogos: {sum}")

    # Carregar e pré-processar o dataset
    dataset = get_dataset(path, (2023, 2024),
                          tier, duration=(30, 120))

    # Mostrar informações do dataset
    print(f"\nDataset pré-processado:")
    print(f"- Número de matches: {len(dataset)}")
    print(f"- Número de features: {len(dataset.columns)}")
    print(f"- Colunas: {dataset.columns[:10]}...")  # Primeiras 10 colunas

    sample = dataset.sample()
    print(f"\nExibindo dados do match_id {sample["match_id"][0]}:")
    print(sample)

    # Salvar dataset processado
    output_path = "./tmp/processed_dataset.json"
    save_dataset(dataset.head(5), output_path)
