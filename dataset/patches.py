import polars as pl

patches = {
    48: 5453,
    49: 9756,
    47: 4246,
    51: 18289,
    50: 13468,
    54: 13265,
    52: 9352,
    53: 9915,
    57: 653,
    55: 6133,
    56: 10844
}


def get_patches(path: str):
    patch_count: dict[int, int] = {}
    for _year in range(2021, 2025):
        meta = pl.scan_csv(
            f"{path}/{_year}/main_metadata.csv").drop_nans("match_id")
        patch_meta = (
            meta.select("match_id", "patch")
            .group_by("patch")
            .agg(pl.count("match_id").alias("count"))
            .select("patch", "count").collect()
        )
        for row in patch_meta.rows(named=True):
            print(row)
            _patch = int(row["patch"])
            _count = int(row["count"])
            patch_count.setdefault(_patch, 0)
            patch_count[_patch] = patch_count[_patch] + _count
    return patch_count
