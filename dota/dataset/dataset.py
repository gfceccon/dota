import ast
import gc
import os
from typing import Optional
import kagglehub
import polars as pl
from dota.logger import LogLevel, get_logger
import dota.dataset.headers as cols
import dota.dataset.schemas as schema
log = get_logger('dataset', LogLevel.INFO, False)


LEAGUES = f"Constants/Constants.Leagues.csv"
HEROES = f"Constants/Constants.Heroes.csv"
PATCHES = f"Constants/Constants.Patch.csv"


def METADATA(x): return f"{x}/main_metadata.csv"
def OBJECTIVES(x): return f"{x}/objectives.csv"
def PICKS_BANS(x): return f"{x}/picks_bans.csv"
def PLAYERS(x): return f"{x}/players.csv"
def EXP_ADV(x): return f"{x}/radiant_exp_adv.csv"
def GOLD_ADV(x): return f"{x}/radiant_gold_adv.csv"
def TEAM_FIGHTS(x): return f"{x}/teamfights.csv"


class Dataset:

    def __init__(self,
                 duration: tuple[int, int] = (10 * 60, 120 * 60),
                 tier: list[str] = ['professional', 'premium'],
                 years: tuple[int, int] = (2020, 2025),
                 dataset: Optional[
                     tuple[
                         pl.LazyFrame,  # Dataset
                         pl.LazyFrame,  # Heroes
                         pl.LazyFrame,  # Patches
                         pl.LazyFrame,  # Metadata
                         pl.LazyFrame,  # Leagues
                     ]
                 ] = None):

        self.data_path = "dota2_data"
        self.dataset_name = "bwandowando/dota-2-pro-league-matches-2023"
        self.path = kagglehub.dataset_download(self.dataset_name)

        self.years = years
        self.duration = duration
        self.tier = tier

        _all_heroes = self._get_lf(HEROES, cols.heroes)
        self.attributes = (
            _all_heroes.select(pl.col("primary_attr")).unique()
            .collect().to_dict(as_series=False)["primary_attr"]
        )
        self.dict_attributes = {attr: i for i,
                                attr in enumerate(self.attributes)}

        self.roles: list[str] = list({
            role for roles_list in
            _all_heroes.select("roles")
            .collect().to_series().to_list()
            for role
            in (ast.literal_eval(roles_list)
                if isinstance(roles_list, str) else roles_list)
        })

        self.dict_roles = {role: i for i, role in enumerate(self.roles)}
        self.roles_idx = [i for i in self.dict_roles.values()]

        self.hero_ids = _all_heroes.select(pl.col("id")).unique().sort(
            "id").collect().to_series().to_list()
        self.dict_hero_index = {hid: i for i, hid in enumerate(self.hero_ids)}

        if (dataset is not None):
            self.data, self.heroes, self.patches, self.metadata, self.leagues = dataset
            if not isinstance(self.data, pl.LazyFrame):
                raise TypeError("Dataset must be a LazyFrame")
            if not isinstance(self.heroes, pl.LazyFrame):
                raise TypeError("Heroes must be a LazyFrame")
            if not isinstance(self.patches, pl.LazyFrame):
                raise TypeError("Patches must be a LazyFrame")
            if not isinstance(self.metadata, pl.LazyFrame):
                raise TypeError("Metadata must be a LazyFrame")
            if not isinstance(self.leagues, pl.LazyFrame):
                raise TypeError("Leagues must be a LazyFrame")
        else:
            self.data = None
            self.heroes = self._heroes()
            self.patches = self._patches()
            self.metadata = self._metadata()
            self.leagues = self._leagues()

    def get_heroes_usage(self, year: int) -> pl.LazyFrame:
        picks_bans = self._picks_bans(year).with_columns(
            pl.col("hero_id").cast(pl.Int32).alias("hero_id"))
        lf = (
            self
            ._metadata()
            .join(self.patches, on="patch", how="inner")
            .filter(pl.col("date").cast(pl.Datetime).dt.year().eq(year))
            .join(self.leagues, on="leagueid", how="inner")
            .join(picks_bans, on="match_id", how="inner")
            .join(self.heroes, on="hero_id", how="inner")
            .select(
                [
                    pl.col("leaguename").alias("league_name"),
                    pl.col("leagueid").alias("league_id"),
                    pl.col("is_pick").alias("pick"),
                    pl.col("patch"),
                    pl.col("date"),
                    pl.col("tier"),
                    pl.col("hero_id"),
                    pl.col("hero_name"),
                ]
            )
        )

        return lf

    def get_year(self, year: int) -> pl.LazyFrame:
        if (self.years[0] > year or self.years[1] <= year):
            raise ValueError(f"Year {year} is not in the range {self.years}")
        metadata = (
            self
            ._get_lf(METADATA(str(year)), cols.metadata)
            .filter(pl.col("duration").is_between(self.duration[0], self.duration[1]))
        )
        objectives = self._objectives(year)
        games = self._games(year)
        exp_adv = self._exp_adv(year)
        gold_adv = self._gold_adv(year)
        team_fights = self._team_fights(year)

        data = (
            metadata
            .join(self.patches, on="patch", how="inner")
            .join(self.leagues, on="leagueid", how="inner")
            .join(games, on=["match_id"], how="inner")
            .join(self.heroes, on="hero_id", how="inner")
            .group_by("match_id")
            .agg([
                pl.all().exclude(*["count", *cols.patches, *
                                   cols.leagues, *cols.heroes, *cols.metadata]),
                *[pl.col(_col).first() for _col in
                  set(col for col in [*cols.patches, *cols.leagues, *cols.metadata] if col not in ["match_id",])],
                pl.when(pl.col("team").eq(0) & pl.col("is_pick").eq(True)).then(
                    pl.col("hero_id")).drop_nulls().alias("radiant_picks"),
                pl.when(pl.col("team").eq(1) & pl.col("is_pick").eq(True)).then(
                    pl.col("hero_id")).drop_nulls().alias("dire_picks"),

                pl.when(pl.col("team").eq(0) & pl.col("is_pick").eq(False)).then(
                    pl.col("hero_id")).drop_nulls().alias("radiant_bans"),
                pl.when(pl.col("team").eq(1) & pl.col("is_pick").eq(False)).then(
                    pl.col("hero_id")).drop_nulls().alias("dire_bans"),
            ])
            .join(objectives, on="match_id", how="inner")
            .join(exp_adv, on="match_id", how="inner")
            .join(gold_adv, on="match_id", how="inner")
        )
        return data

    def _patches(self) -> pl.LazyFrame:
        patch_count: dict[int, tuple[str, str, int]] = {}
        patches_detail = self._get_lf(PATCHES, cols.patches).collect()
        meta = self._metadata().drop_nans("match_id")
        patch_meta = (
            meta.select("match_id", "patch")
            .group_by("patch")
            .agg(pl.count("match_id").alias("count"))
            .select("patch", "count").collect()
        )
        for row in patch_meta.iter_rows(named=True):
            patch = row["patch"]
            count = row["count"]
            patch_name = patches_detail.item(patch - 1, 0)
            patch_date = patches_detail.item(patch - 1, 1)
            patch_count[patch] = (patch_name, patch_date, count)
        return pl.LazyFrame(
            {
                "patch": list(patch_count.keys()),
                "name": [x[0] for x in patch_count.values()],
                "date": [x[1] for x in patch_count.values()],
                "count": [x[2] for x in patch_count.values()],
            }
        ).sort("patch")

    def _leagues(self) -> pl.LazyFrame:
        leagues = (
            self._get_lf(LEAGUES, cols.leagues)
            .filter(pl.col("tier").is_in(self.tier))
        )
        return leagues

    def _heroes(self, ):
        heroes = (
            self._get_lf(HEROES, cols.heroes)
            .with_columns(
                pl.col("primary_attr").map_elements(lambda x: self.dict_attributes.get(x) if isinstance(
                    x, str) else x, return_dtype=pl.Int32).alias("primary_attribute"),
                pl.col("roles").map_elements(
                    lambda x: [self.dict_roles.get(y) for y in ast.literal_eval(
                        x)] if isinstance(x, str) else x,
                    return_dtype=pl.List(pl.Int32)
                ),
                pl.col("attack_type").map_elements(
                    lambda x: 0 if x == "Melee" else 1 if x == "Ranged" else None, return_dtype=pl.UInt8
                ).alias("attack_type"),
                pl.col("id")
                .replace(self.dict_hero_index)
                .cast(pl.Int32)
                .alias("hero_id"),
            )
            .with_columns(
                pl.col("roles").map_elements(
                    lambda x: [1 if i in x else 0 for i in self.roles_idx],
                    return_dtype=pl.List(pl.Int32)
                ).alias("roles_vector"),
                pl.col("localized_name").alias("hero_name"),
            )
        )
        return heroes

    def _picks_bans(self, year: int) -> pl.LazyFrame:
        picks_bans = self._get_lf(PICKS_BANS(str(year)), cols.picks_bans)
        picks = (
            picks_bans
            .select(
                [
                    pl.col("team"),
                    pl.col("match_id"),
                    pl.col("is_pick"),
                    pl.col("hero_id")
                    .cast(pl.Int32)
                ]
            )
        )
        return picks

    def _objectives(self, year: int) -> pl.LazyFrame:
        obj = self._get_lf(OBJECTIVES(str(year)), cols.objectives)

        objectives = (
            obj
            .with_columns([
                pl.col("type")
                .replace(cols.objectives_type)
            ])
            .group_by(["match_id"])
            .agg(
                pl.concat_list([
                    pl.col(f"{col}")
                    .drop_nulls()
                    for col in cols.objectives if col not in ["match_id"]
                ]).alias("objectives")
            )
        )
        return objectives

    def _games(self, year: int) -> pl.LazyFrame:
        players = self._get_lf(PLAYERS(str(year)), cols.players)

        players = (
            players
            .select(cols.players)
            .with_columns(
                pl.col("hero_id").cast(pl.Int32).alias("hero_id"),)
            .join(self._picks_bans(year), on=["match_id", "hero_id"], how="right")
        )

        return players

    def _exp_adv(self, year: int) -> pl.LazyFrame:
        return self._get_lf(EXP_ADV(str(year)), cols.exp_adv).select(pl.col("exp").alias("exp_adv"), pl.col("match_id")).group_by("match_id").agg(pl.all())

    def _gold_adv(self, year: int) -> pl.LazyFrame:
        return self._get_lf(GOLD_ADV(str(year)), cols.gold_adv).select(pl.col("gold").alias("gold_adv"), pl.col("match_id")).group_by("match_id").agg(pl.all())

    def _team_fights(self, year: int) -> pl.LazyFrame:
        return self._get_lf(TEAM_FIGHTS(str(year)), cols.team_fights).group_by("match_id").agg(pl.all())

    def _metadata(self) -> pl.LazyFrame:
        files = [METADATA(str(year))
                 for year in range(self.years[0], self.years[1])]
        lf, lost = self._get_all_lf(*files)
        log.debug(f"Difference in metadata columns: {lost}")
        return lf

    def _get_all_lf(self, *files: str) -> tuple[pl.LazyFrame, list[str]]:
        scans: list[pl.LazyFrame] = []
        schemas: list[pl.Schema] = []
        for file in files:
            _lf = pl.scan_csv(f"{self.path}/{file}")
            scans.append(_lf)
            schemas.append(_lf.collect_schema())
        names = set(schemas[0].names())
        lost = []

        for schema in schemas:
            names.intersection_update(schema.names())
            lost.extend(set(schema.names()) - names)
        lf = pl.concat(scans, how="diagonal_relaxed").lazy()
        return lf, lost

    def _get_lf(self, file: str, columns: list[str]) -> pl.LazyFrame:
        lf = pl.scan_csv(f"{self.path}/{file}")
        lf = (
            lf.select(_col for _col in columns)
        )
        return lf

    def save_dataset(self, year: int, head: Optional[int] = None):
        if year is not None:
            if not (self.years[0] <= year < self.years[1]):
                raise ValueError(
                    f"Year {year} is not in the range {self.years}")
        os.makedirs(self.data_path, exist_ok=True)
        path = f"{self.data_path}/{year}"
        os.makedirs(path, exist_ok=True)

        if head is None:
            df = self.get_year(year).collect()
        else:
            df = self.get_year(year).filter(
                pl.col("leaguename")
                .eq(f"The International {year}")).head(head).collect()
            


        df = df.select(
            *[
                pl.col({col}).list.drop_nulls() 
                for col in schema.dataset_schema.names() 
                if schema.dataset_schema[col].is_nested()
            ],
            *[
                pl.col({col}).drop_nulls()
                for col in schema.dataset_schema.names() 
                if not schema.dataset_schema[col].is_nested()
            ],
        )

        with open(f"{path}/dataset_schema.txt", "w") as f:
            f.write(str(df.collect_schema()))
        df.write_json(f"{path}/dataset.json")

        print(f"Year {year} matches:", df.shape[0])

        df = self._objectives(year).collect()
        with open(f"{path}/objectives_schema.txt", "w") as f:
            f.write(str(df.collect_schema()))
        df.write_json(f"{path}/objectives.json")

        df = self._exp_adv(year).collect()
        with open(f"{path}/xp_adv_schema.txt", "w") as f:
            f.write(str(df.collect_schema()))
        df.write_json(f"{path}/xp_adv.json")

        df = self._gold_adv(year).collect()
        with open(f"{path}/gold_adv_schema.txt", "w") as f:
            f.write(str(df.collect_schema()))
        df.write_json(f"{path}/gold_adv.json")

        df = self._team_fights(year).collect()
        with open(f"{path}/team_fights_schema.txt", "w") as f:
            f.write(str(df.collect_schema()))
        df.write_json(f"{path}/team_fights.json")

        df = self._picks_bans(year).collect()
        with open(f"{path}/picks_bans_schema.txt", "w") as f:
            f.write(str(df.collect_schema()))
        df.write_json(f"{path}/picks_bans.json")

    def save_metadata(self):
        os.makedirs(self.data_path, exist_ok=True)

        df = self.heroes.collect()
        with open(f"{self.data_path}/heroes_schema.txt", "w") as f:
            f.write(str(df.collect_schema()))
        df.write_json(f"{self.data_path}/heroes.json")

        df = self.metadata.collect()
        with open(f"{self.data_path}/metadata_schema.txt", "w") as f:
            f.write(str(df.collect_schema()))
        df.write_json(f"{self.data_path}/metadata.json")

        df = self.leagues.collect()
        with open(f"{self.data_path}/leagues_schema.txt", "w") as f:
            f.write(str(df.collect_schema()))
        df.write_json(f"{self.data_path}/leagues.json")

        df = self.patches.collect()
        with open(f"{self.data_path}/patches_schema.txt", "w") as f:
            f.write(str(df.collect_schema()))
        df.write_json(f"{self.data_path}/patches.json")

    @staticmethod
    def load(path: str, year: int) -> tuple[
            pl.LazyFrame,
            pl.LazyFrame,
            pl.LazyFrame,
            pl.LazyFrame,
            pl.LazyFrame,]:
        if not (2020 <= year < 2025):
            raise ValueError("Year must be between 2020 and 2024")
        _path = f"{path}/dota2_data/{year}/dataset.json"
        if not os.path.exists(_path):
            raise FileNotFoundError(
                f"Dataset for year {year} not found at {_path}")
        return (
            pl.scan_ndjson(_path, schema=schema.dataset_schema),  # Dataset
            pl.scan_ndjson(f"{path}/dota2_data/heroes.json",
                           schema=schema.heroes_schema),  # Heroes
            pl.scan_ndjson(f"{path}/dota2_data/patches.json",
                           schema=schema.patches_schema),  # Patches
            pl.scan_ndjson(f"{path}/dota2_data/metadata.json",
                           schema=schema.metadata_schema),  # Metadata
            pl.scan_ndjson(f"{path}/dota2_data/leagues.json",
                           schema=schema.leagues_schema),  # Leagues
        )
