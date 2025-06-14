from dota import Dota2
from dota import Dataset
import polars as pl

if __name__ == "__main__":
    dataset = Dataset()
    lf = dataset.get_year(2023)
    # print(lf.collect_schema())
    # print(lf.explain(optimized=True))
    #print(lf.head(5).collect())
    for year in range(2020, 2025):
        usage = dataset.get_heroes_usage(year)
        # lf = dataset.get_year(year)
        df_usage = usage.collect()
        # df = lf.collect()
        print(f"Year: {year}")
        hero_usage = (
            df_usage
            .group_by(["patch"])
            .agg([
                pl.col("hero_id").count().alias("hero_count"),
                pl.when(pl.col("pick") == True).then(pl.col("hero_id").count()).otherwise(0).alias("picks"),
                pl.when(pl.col("pick") == False).then(pl.col("hero_id").count()).otherwise(0).alias("bans"),
            ])
        )

        usage = None
        df_usage = None
        # lf = None
        # df = None