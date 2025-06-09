import kagglehub
import polars as pl
from files import (
    Dota2Files,
    get_lf,
)
from patches import get_patches


def get_best_leagues(path: str, min_duration=10 * 60, max_duration=120 * 60):
    patches_detail = get_lf(Dota2Files.PATCHES, path).collect()
    asd = []
    for idx, row in enumerate(patches_detail.iter_rows(named=True)):
        date = row["date"]
        patch_name = patches_detail.item(idx, 0)
        asd.append((idx, patch_name, date))

    dsa = pl.LazyFrame(asd, schema=["patch", "name", "date"], orient="row")

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
            other=dsa,
            left_on="patch",
            right_on="patch",
            how="left"
        )
        .filter(
            pl.col("duration").is_between(min_duration, max_duration),
            pl.col("radiant_win").is_not_null(),
        )
        .select([
            "match_id",
            "league_id",
            "league_name",
            "patch",
            "date",
        ])
    )
    return matches.group_by("league_name").agg(
        pl.count("match_id").alias("match_count"),
        pl.col("patch").unique().alias("patches"),
        pl.col("date").unique()
    ).sort("match_count", descending=True).collect()


if __name__ == "__main__":
    # Download the dataset
    dataset_name = "bwandowando/dota-2-pro-league-matches-2023/versions/177"
    path = kagglehub.dataset_download(dataset_name)

    patch = 56
    tournaments = ["The International", "DPC", "Major", "ESL", "PGL", "DreamLeague", "BetBoom", "Blast", "Riyadh Masters", "Invitational"]
    asd = get_best_leagues(path).filter(
        pl.col("league_name").str.contains_any(tournaments),
        pl.col("patches").list.contains(patch)
    ).sort("match_count", descending=True).select(pl.col("match_count")).sum()
    print(asd)
