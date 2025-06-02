import kagglehub
import polars as pl
from src.files import (
    get_metadata,
    patches_file,
)


def get_patches(path: str, begin_year: int = 2021, end_year: int = 2024) -> dict[int, tuple[int, str]]:
    patch_count: dict[int, tuple[int, str]] = {}
    patches_detail = pl.scan_csv(f"{path}/{patches_file}").collect()
    for _year in range(begin_year, end_year + 1):
        meta = pl.scan_csv(
            f"{path}/{get_metadata(str(_year))}").drop_nans("match_id")
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
        patch_count[_idx] = patch_count[_idx][0], patches_detail.item(_patch, "patch")
    return patch_count

if __name__ == "__main__":
    # Download do dataset
    dataset_name = "bwandowando/dota-2-pro-league-matches-2023/versions/177"
    path = kagglehub.dataset_download(dataset_name)

    print(f"Patches:")
    print(get_patches(path))