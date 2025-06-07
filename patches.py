import kagglehub
import polars as pl
from files import (
    get_lf,
    Dota2Files,
)


def get_patches(path: str, year: int = 2021) -> dict[int, tuple[int, str]]:
    patch_count: dict[int, tuple[int, str]] = {}
    patches_detail = get_lf(Dota2Files.PATCHES, path).collect()
    meta = get_lf(Dota2Files.METADATA, path, year).drop_nans("match_id")
    patch_meta = (
        meta.select("match_id", "patch")
        .group_by("patch")
        .agg(pl.count("match_id").alias("count"))
        .select("patch", "count").collect()
    )
    for row in patch_meta.rows(named=True):
        _patch = int(row["patch"])
        _count = int(row["count"])
        patch_count.setdefault(_patch, (0, ""))
        patch_count[_patch] = patch_count[_patch][0] + _count, ""
    for _patch, _idx in enumerate(patch_count):
        patch_count[_idx] = patch_count[_idx][0], patches_detail.item(
            _patch, "patch")
    return patch_count


if __name__ == "__main__":
    # Download do dataset
    dataset_name = "bwandowando/dota-2-pro-league-matches-2023/versions/177"
    path = kagglehub.dataset_download(dataset_name)

    print(f"Patches:")
    print(get_patches(path))
