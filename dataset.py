import polars as pl
from patches import get_patches
from matches import get_matches
from players import get_players_draft


def preprocess_dataset(path: str, patches: list[int], tier: list[str],
                       min_duration: int = 10 * 60, max_duration: int = 120 * 60) -> tuple[pl.LazyFrame, list[str], list[str]]:

    # Carregando e filtrando partidas
    matches = get_matches(path, patches, tier, min_duration, max_duration)

    # Carregando jogos e draft de jogadores
    games, players_cols, hero_cols = get_players_draft(path, matches)

    # Agrupando dados por partida
    dataset = (
        games
        .group_by("match_id")
        .agg(
            pl.all().drop_nulls(),
            pl.when(pl.col("team").eq(0) & pl.col("pick").eq(True)).then(
                pl.col("hero_id") + 1).drop_nulls().alias("radiant_picks"),
            pl.when(pl.col("team").eq(1) & pl.col("pick").eq(True)).then(
                pl.col("hero_id") + 1).drop_nulls().alias("dire_picks"),

            pl.when(pl.col("team").eq(0) & pl.col("pick").eq(False)).then(
                pl.col("hero_id") + 1).drop_nulls().alias("radiant_bans"),
            pl.when(pl.col("team").eq(1) & pl.col("pick").eq(False)).then(
                pl.col("hero_id") + 1).drop_nulls().alias("dire_bans"),

            *[
                pl.when(pl.col("team").eq(team_id) &
                        pl.col("pick").eq(True))
                .then(
                    pl.concat_list(
                        [
                            (pl.col(f"{stat}") * 1.0 - pl.col(f"{stat}").min()) /
                            pl.when((pl.col(f"{stat}").max() - pl.col(f"{stat}").min()) != 0)
                            .then(pl.col(f"{stat}").max() - pl.col(f"{stat}").min())
                            .otherwise(1.0)
                            for stat in players_cols]
                    ))
                .drop_nulls()
                .drop_nans()
                .alias(f"{team_name}_stats_normalized")
                for team_id, team_name in [(0, "radiant"), (1, "dire")]
            ],

        )
        .filter(
            (~pl.col("radiant_stats_null").list.any()) &
            (pl.col("radiant_picks").list.len() == 5) &
            (pl.col("radiant_bans").list.len() == 7) &
            (~pl.col("dire_stats_null").list.any()) &
            (pl.col("dire_picks").list.len() == 5) &
            (pl.col("dire_bans").list.len() == 7)
        )
        # .with_columns(
        #     pl.when(pl.col("radiant_stats").is_not_null())
        #     .then(
        #         pl.col("radiant_stats").list.eval(
        #             (pl.element() - pl.col("min_stats")) / (pl.col("max_stats") - pl.col("min_stats"))
        #         ))
        #     .alias("radiant_stats_normalized"),
        #     pl.when(pl.col("dire_stats").is_not_null())
        #     .then(
        #         pl.col("dire_stats").list.eval(
        #             (pl.element() - pl.col("min_stats")) / (pl.col("max_stats") - pl.col("min_stats"))
        #         ))
        #     .alias("dire_stats_normalized"),
        # )
    )

    return dataset, players_cols, hero_cols


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
            "match_id",
            "radiant_hero_roles", "dire_hero_roles",
            "radiant_stats", "dire_stats",
            "radiant_picks", "dire_picks",
            "radiant_bans", "dire_bans",
            #"min_stats", "max_stats",
            "radiant_stats_normalized", "dire_stats_normalized"
        )
        .collect())
    print("Dataset carregado e pré-processado com sucesso!")
    return dataset, games_cols, hero_cols


def save_dataset(dataset: pl.DataFrame, output_path: str = "./tmp/DATASET.json") -> None:
    print(f"Salvando dataset em {output_path}...")
    dataset.write_json(output_path)
    print("Dataset salvo com sucesso!")
