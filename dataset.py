import kagglehub
import polars as pl
from patches import get_patches
from matches import get_matches
from players import get_players_draft


from typing import Optional

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
            
            pl.when(pl.col("team").eq(0) & pl.col("pick").eq(True)).then(
                pl.col("hero_idx") + 1).drop_nulls().alias("radiant_picks_idx"),
            pl.when(pl.col("team").eq(1) & pl.col("pick").eq(True)).then(
                pl.col("hero_idx") + 1).drop_nulls().alias("dire_picks_idx"),

            pl.when(pl.col("team").eq(0) & pl.col("pick").eq(False)).then(
                pl.col("hero_idx") + 1).drop_nulls().alias("radiant_bans_idx"),
            pl.when(pl.col("team").eq(1) & pl.col("pick").eq(False)).then(
                pl.col("hero_idx") + 1).drop_nulls().alias("dire_bans_idx"),

            *[
                pl.when(pl.col("team").eq(team_id) &
                        pl.col("pick").eq(True))
                .then(
                    (pl.col(f"{stat}") * 1.0 - pl.col(f"{stat}").min()) /
                    pl.when(
                        (pl.col(f"{stat}").max() - pl.col(f"{stat}").min()) != 0)
                    .then(pl.col(f"{stat}").max() - pl.col(f"{stat}").min())
                    .otherwise(1.0)
                )
                .fill_null(0)
                .alias(f"{team_name}_{stat}")
                for stat in players_cols
                for team_id, team_name in [(0, "radiant"), (1, "dire")]
            ],

            *[
                pl.when(pl.col("team").eq(team_id) &
                        pl.col("pick").eq(True))
                .then(
                    (pl.col(f"{stat}") * 1.0 - pl.col(f"{stat}").min()) /
                    pl.when(
                        (pl.col(f"{stat}").max() - pl.col(f"{stat}").min()) != 0)
                    .then(pl.col(f"{stat}").max() - pl.col(f"{stat}").min())
                    .otherwise(1.0)
                )
                .fill_null(0)
                .alias(f"{team_name}_hero_{stat}")
                for stat in hero_cols
                for team_id, team_name in [(0, "radiant"), (1, "dire")]
            ],
            
            pl.col("league_id").first().alias("league"),
        )
        .drop("league_id")
        .with_columns(
            pl.col("league").alias("league_id"),)
        .filter(
            (~pl.col("radiant_stats_null").list.any()) &
            (~pl.col("dire_stats_null").list.any()) &

            (pl.col("radiant_picks").list.len() == 5) &
            (pl.col("dire_picks").list.len() == 5) &

            (pl.col("radiant_bans").list.len() == 7) &
            (pl.col("dire_bans").list.len() == 7)
        )
    )

    return dataset, players_cols, hero_cols


def get_dataset(
        path: str,
        tier: list[str] = ['professional'],
        duration: tuple[int, int] = (30, 120),
        specific_patches: list[int] = [], verbose: bool = True
) -> tuple[pl.DataFrame, list[str], list[str]]:
    if verbose:
        print(f"Carregando dataset...")
        print(f"Tier: {tier}, Duração: {duration[0]}-{duration[1]} minutos")
    patches = get_patches(path)
    if verbose:
        if specific_patches:
            print(f"Patches: {",".join([f'{patches[patch_id][1]} ({patches[patch_id][0]})' for patch_id in specific_patches])}")
        else:
            print(f"Patches: {",".join([f'{patches[patch_id][1]} ({patches[patch_id][0]})' for patch_id in patches.keys()])}")

    dataset, player_cols, hero_cols = preprocess_dataset(
        path,
        list(patches.keys()) if not specific_patches else specific_patches,
        tier,
        duration[0] * 60,
        duration[1] * 60
    )

    dataset = (
        dataset
        .with_columns(
            *[pl.col(f"{team_name}_{stat}").list.get(i).alias(f"{team_name}_{stat}_{i}")
                for stat in player_cols
                for team_name in ["radiant", "dire"]
                for i in range(5)
              ],
            *[pl.col(f"{team_name}_hero_{stat}").list.get(i).alias(f"{team_name}_hero_{stat}_{i}")
                for stat in hero_cols
                for team_name in ["radiant", "dire"]
                for i in range(5)
              ],
        )
        .with_columns(
            *[pl.concat_list([pl.col(f"{team_name}_{stat}_{i}")
                        for stat in player_cols
                         for i in range(5)
                         ]).alias(f"{team_name}_features",)
                for team_name in ["radiant", "dire"]],


            *[pl.concat_list([pl.col(f"{team_name}_hero_{stat}_{i}")
                        for stat in hero_cols
                         for i in range(5)
                         ]).alias(f"{team_name}_hero_features",)
                for team_name in ["radiant", "dire"]],
        )
        .select(
            "match_id",
            "radiant_hero_roles", "dire_hero_roles",
            
            "radiant_picks", "dire_picks",
            "radiant_bans", "dire_bans",
            
            "radiant_picks_idx", "dire_picks_idx",
            "radiant_bans_idx", "dire_bans_idx",
            
            "radiant_features", "dire_features",
            "radiant_hero_features", "dire_hero_features",
            "league_id",
        )
        .collect()
    )
    return dataset, player_cols, hero_cols


if __name__ == "__main__":
    dataset_name = "bwandowando/dota-2-pro-league-matches-2023"
    path = kagglehub.dataset_download(dataset_name)

    matches, _ = get_matches(path, patches=[54], tier=[
                          'professional'], min_duration=30 * 60, max_duration=120 * 60)
    games, players_cols, hero_cols = get_players_draft(path, matches)

    print(f"Schema: {[x for x, y in zip(games.collect_schema().names(), games.collect_schema().dtypes()) if y.is_numeric() == False]}")
