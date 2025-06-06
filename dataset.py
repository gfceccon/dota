import ast
import kagglehub
import polars as pl
from patches import get_patches
from matches import get_matches
from players import get_players_draft
import ast


def preprocess_dataset(path: str, patches: list[int], tier: list[str],
                       min_duration: int = 10 * 60, max_duration: int = 120 * 60) -> tuple[pl.LazyFrame, list[str], list[str]]:

    # Carregando e filtrando partidas
    matches = get_matches(path, patches, tier, min_duration, max_duration)

    # Carregando jogos e draft de jogadores
    games, games_cols, hero_cols = get_players_draft(path, matches)

    # Agrupando dados por partida
    dataset = (
        games.group_by("match_id")
        .agg(
            pl.all().drop_nulls(),
            pl.when(pl.col("team") == 0, pl.col("pick")).then(
                pl.col("hero_id")).drop_nulls().alias("radiant_picks"),
            pl.when(pl.col("team") == 1, pl.col("pick")).then(
                pl.col("hero_id")).drop_nulls().alias("dire_picks"),

            pl.when(pl.col("team") == 0, ~pl.col("pick")).then(
                pl.col("hero_id")).drop_nulls().alias("radiant_bans"),
            pl.when(pl.col("team") == 1, ~pl.col("pick")).then(
                pl.col("hero_id")).drop_nulls().alias("dire_bans"),

            pl.concat_list([pl.col(f"{col}").max()
                           for col in games_cols]).alias("max_stats"),
            
            pl.concat_list([pl.col(f"{col}").min()
                           for col in games_cols]).alias("min_stats"),
        ))
    return dataset, games_cols, hero_cols


def get_dataset(path: str, years: tuple[int, int] = (2023, 2024), tier: list[str] = ['professional'], duration: tuple[int, int] = (30, 120)) -> tuple[pl.DataFrame, list[str], list[str]]:
    print(f"Carregando dataset de {years[0]} a {years[1]}...")
    print(f"Tier: {tier}, Duração: {duration[0]}-{duration[1]} minutos")

    patches = get_patches(path, begin_year=years[0], end_year=years[1])
    dataset, games_cols, hero_cols = preprocess_dataset(
        path,
        list(patches.keys()),
        tier,
        duration[0] * 60,
        duration[1] * 60
    )

    dataset = (
        dataset
        .select(
            "radiant_hero_roles", "dire_hero_roles",
            "radiant_stats", "dire_stats",
            "radiant_picks", "dire_picks",
            "radiant_bans", "dire_bans",
            "min_stats", "max_stats",)
        .collect())
    print("Dataset carregado e pré-processado com sucesso!")
    return dataset, games_cols, hero_cols


def save_dataset(dataset: pl.DataFrame, output_path: str = "./tmp/DATASET.json") -> None:
    print(f"Salvando dataset em {output_path}...")
    dataset.write_json(output_path)
    print("Dataset salvo com sucesso!")
