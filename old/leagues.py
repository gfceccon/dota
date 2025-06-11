import kagglehub
import polars as pl
from files import (
    Dota2Files,
    get_lf,
)
from patches import get_patches


def get_tier_one(path: str, tournaments: list[str] = [], default_tournaments: list[str] =
                 ["The International", "DPC", "Major", "ESL", "PGL",
                  "DreamLeague", "BetBoom", "Blast", "Riyadh Masters", "Invitational"],
                 years: tuple[int, int] = (2020, 2025)) -> pl.DataFrame:

    tournaments = tournaments + default_tournaments
    patches_detail = get_lf(Dota2Files.PATCHES, path).collect()
    patches_list = []
    for idx, row in enumerate(patches_detail.iter_rows(named=True)):
        date = row["date"]
        patch_name = patches_detail.item(idx, 0)
        patches_list.append((idx, patch_name, date))

    patches = pl.LazyFrame(patches_list, schema=["patch", "name", "date"], orient="row")

    leagues = (
        get_lf(Dota2Files.LEAGUES, path)
        .select([
            pl.col("leagueid").alias("league_id"),
            pl.col("leaguename").alias("league_name"),
            pl.col("tier").alias("league_tier")
        ])
    )

    matches = (
        get_lf(Dota2Files.METADATA, path, (2020, 2025))
        .drop_nans(["match_id", "leagueid"])
        .with_columns(pl.col("leagueid").alias("league_id"))
        .join(other=leagues, on="league_id", how="left")
        .join(
            other=patches,
            left_on="patch",
            right_on="patch",
            how="left"
        )
    )

    return matches.group_by(["league_id", "patch"]).agg(
        pl.count("match_id").alias("match_count"),
        pl.col("league_name").first().alias("league_name"),
    ).filter(pl.col("league_name").str.contains_any(tournaments),).collect()


if __name__ == "__main__":
    # Download the dataset
    dataset_name = "bwandowando/dota-2-pro-league-matches-2023/versions/177"
    path = kagglehub.dataset_download(dataset_name)
    asd = get_tier_one(path).filter(
        pl.col("league_name").str.contains("The International")).sort("match_count", descending=True)
    print(asd)
