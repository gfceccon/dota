import kagglehub
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
                            pl.when(
                                (pl.col(f"{stat}").max() - pl.col(f"{stat}").min()) != 0)
                            .then(pl.col(f"{stat}").max() - pl.col(f"{stat}").min())
                            .otherwise(1.0)
                            for stat in players_cols]
                    ))
                .drop_nulls()
                .drop_nans()
                .alias(f"{team_name}_stats_normalized")
                for team_id, team_name in [(0, "radiant"), (1, "dire")]
            ],

            *[
                pl.when(pl.col("team").eq(team_id) &
                        pl.col("pick").eq(True))
                .then(
                    pl.concat_list(
                        [
                            (pl.col(f"{stat}") * 1.0 - pl.col(f"{stat}").min()) /
                            pl.when(
                                (pl.col(f"{stat}").max() - pl.col(f"{stat}").min()) != 0)
                            .then(pl.col(f"{stat}").max() - pl.col(f"{stat}").min())
                            .otherwise(1.0)
                            for stat in hero_cols]
                    ))
                .drop_nulls()
                .drop_nans()
                .alias(f"{team_name}_hero_stats_normalized")
                for team_id, team_name in [(0, "radiant"), (1, "dire")]
            ],

        )
        .filter(
            (~pl.col("radiant_stats_null").list.any()) &
            (~pl.col("dire_stats_null").list.any()) &

            (pl.col("dire_hero_stats_normalized").list.len() == 5) &
            (pl.col("radiant_hero_stats_normalized").list.len() == 5) &


            (pl.col("radiant_picks").list.len() == 5) &
            (pl.col("dire_picks").list.len() == 5) &

            (pl.col("radiant_bans").list.len() == 7) &
            (pl.col("dire_bans").list.len() == 7)
        )
    )

    return dataset, players_cols, hero_cols


def get_dataset(
        path: str, year: int = 2024,
        tier: list[str] = ['professional'],
        duration: tuple[int, int] = (30, 120),
        specific_patches: list[int] = []
) -> tuple[pl.DataFrame, list[str], list[str]]:
    print(f"Carregando dataset...")
    print(f"Tier: {tier}, Duração: {duration[0]}-{duration[1]} minutos")
    patches = get_patches(path, year)
    print("Patches:")
    if specific_patches:
        for patch_id in specific_patches:
            [count, name] = patches.get(patch_id, (0, ""))
            print(f"Patch {name} ({patch_id}): {count} partidas")
    else:
        for patch, (count, name) in patches.items():
            print(f"Patch {name} ({patch}): {count} partidas")

    dataset, games_cols, hero_cols = preprocess_dataset(
        path,
        list(patches.keys()) if not specific_patches else specific_patches,
        tier,
        duration[0] * 60,
        duration[1] * 60
    )

    dataset = (
        dataset
        .select(
            "match_id",
            "radiant_hero_roles", "dire_hero_roles",
            "radiant_picks", "dire_picks",
            "radiant_bans", "dire_bans",
            "radiant_stats_normalized", "dire_stats_normalized",
            "radiant_hero_stats_normalized", "dire_hero_stats_normalized"
        )
        .collect()
    )
    print("Dataset carregado e pré-processado com sucesso!")
    return dataset, games_cols, hero_cols


def save_dataset(dataset: pl.DataFrame, output_path: str = "./tmp/DATASET.json") -> None:
    print(f"Salvando dataset em {output_path}...")
    dataset.write_json(output_path)
    print("Dataset salvo com sucesso!")


if __name__ == "__main__":
    dataset_name = "bwandowando/dota-2-pro-league-matches-2023"
    path = kagglehub.dataset_download(dataset_name)

    matches = get_matches(path, patches=[54], tier=[
                          'professional'], min_duration=30 * 60, max_duration=120 * 60)
    games, players_cols, hero_cols = get_players_draft(path, matches)

    print(f"Schema: {[x for x, y in zip(games.collect_schema().names(), games.collect_schema().dtypes()) if y.is_numeric() == False]}")
