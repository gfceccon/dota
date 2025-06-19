import ast
import gc
import os
from typing import Optional
import kagglehub
import polars as pl
from dota.logger import get_logger
import dota.dataset.headers as cols
import dota.dataset.schemas as schema
import pandas as pd
import pandas as pd

log = get_logger('dataset')


LEAGUES = f"Constants/Constants.Leagues.csv"
HEROES = f"Constants/Constants.Heroes.csv"
ITEMS = f"Constants/Constants.ItemIDs.csv"
ITEMS = f"Constants/Constants.ItemIDs.csv"
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
                 duration: tuple[int, int] = (10 * 60, 150 * 60),
                 tier: list[str] = ['professional', 'premium'],
                 years: tuple[int, int] = (2021, 2025)):
        self.data_path = "dota2_data"
        self.dataset_name = "bwandowando/dota-2-pro-league-matches-2023"
        
        self.dataset_version = "181"
        self.path = f"~/.cache/kagglehub/datasets/{self.dataset_name}/versions/{self.dataset_version}/"
        if(not os.path.exists(self.path)):
            self.path = kagglehub.dataset_download(self.dataset_name)
            print(f"Dataset downloaded to {self.path}")

        self.years = years
        self.duration = duration
        self.tier = tier

        _all_heroes = self._get_lf(HEROES, cols.heroes)
        self.attributes = (
            _all_heroes.select(pl.col("primary_attr")).unique()
            .collect().to_dict(as_series=False)["primary_attr"]
        )
        self.dict_attributes = {attr: i + 1 for i,
                                attr in enumerate(self.attributes)}
        self.dict_attack = {
            "Melee": 1,
            "Ranged": 2,
            "Unknown": 0
        }

        self.roles: list[str] = list({
            role for roles_list in
            _all_heroes.select("roles")
            .collect().to_series().to_list()
            for role
            in (ast.literal_eval(roles_list)
                if isinstance(roles_list, str) else roles_list)
        })

        self.dict_roles = {role: i + 1 for i, role in enumerate(self.roles)}

        self.hero_ids = _all_heroes.select(pl.col("id")).unique().sort(
            "id").collect().to_series().to_list()
        self.dict_hero_index = {hid: i for i, hid in enumerate(self.hero_ids)}

        _all_items = self._get_lf(ITEMS, cols.items)
        self.items_id = _all_items.select(pl.col("id")).unique().sort(
            "id").collect().to_series().to_list()
        self.dict_items_index = {hid: i + 1 for i,
                                 hid in enumerate(self.items_id)}

        self.heroes = self._heroes()
        self.patches = self._patches()
        self.metadata = self._metadata()
        self.leagues = self._leagues()
        self.items = self._items()

        _all_items = self._get_lf(ITEMS, cols.items)
        self.items_id = _all_items.select(pl.col("id")).unique().sort(
            "id").collect().to_series().to_list()
        self.dict_items_index = {hid: i for i, hid in enumerate(self.items_id)}

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
        # Dados de team_fights ainda não estão sendo utilizados
        # team_fights = self._team_fights(year)

        self.data = (
            metadata
            .join(self.patches, on="patch", how="inner")
            .join(self.leagues, on="leagueid", how="inner")
            .join(games, on=["match_id"], how="inner")
            .join(self.heroes, on="hero_id", how="inner")
            .with_columns(
                # Players Stats
                pl.when(pl.col("team").eq(0) & pl.col("is_pick").eq(True))
                .then(
                    pl.concat_list(
                        pl.when(pl.col(x).is_not_null())
                        .then(pl.col(x))
                        .otherwise(pl.lit(0))
                        for x in cols.players_stats_single
                    ).alias("player_radiant_stats")
                ),
                pl.when(pl.col("team").eq(1) & pl.col("is_pick").eq(True))
                .then(
                    pl.concat_list(
                        pl.when(pl.col(x).is_not_null())
                        .then(pl.col(x))
                        .otherwise(pl.lit(0))
                        for x in cols.players_stats_single
                    ).alias("player_dire_stats")
                ),

                # Items
                pl.when(pl.col("team").eq(0) & pl.col("is_pick").eq(True)).then(
                    pl.col("items_vector")).alias("radiant_items"),

                pl.when(pl.col("team").eq(1) & pl.col("is_pick").eq(True)).then(
                    pl.col("items_vector")).alias("dire_items"),

                # Backpack
                pl.when(pl.col("team").eq(0) & pl.col("is_pick").eq(True)).then(
                    pl.col("backpack_vector")).alias("radiant_backpack"),

                pl.when(pl.col("team").eq(1) & pl.col("is_pick").eq(True)).then(
                    pl.col("backpack_vector")).alias("dire_backpack"),
            )
            .group_by(["match_id"])
            .agg([
                pl.all().exclude(*["count", *cols.patches, *
                                   cols.leagues, *cols.heroes, *cols.metadata]),

                *[pl.col(_col).first() for _col in
                  set(col for col in [*cols.patches, *cols.leagues, *cols.metadata] if col not in ["match_id",])],

                # Picks and Bans
                pl.when(pl.col("team").eq(0) & pl.col("is_pick").eq(True)).then(
                    pl.col("hero_idx")).drop_nulls().alias("radiant_picks"),

                pl.when(pl.col("team").eq(1) & pl.col("is_pick").eq(True)).then(
                    pl.col("hero_idx")).drop_nulls().alias("dire_picks"),

                pl.when(pl.col("team").eq(0) & pl.col("is_pick").eq(False)).then(
                    pl.col("hero_idx")).drop_nulls().alias("radiant_bans"),

                pl.when(pl.col("team").eq(1) & pl.col("is_pick").eq(False)).then(
                    pl.col("hero_idx")).drop_nulls().alias("dire_bans"),

                # Hero Attributes
                pl.when(pl.col("team").eq(0) & pl.col("is_pick").eq(True)).then(
                    pl.col("primary_attribute")).alias("radiant_attributes"),

                pl.when(pl.col("team").eq(1) & pl.col("is_pick").eq(True)).then(
                    pl.col("primary_attribute")).alias("dire_attributes"),

                # Hero Attack type
                pl.when(pl.col("team").eq(0) & pl.col("is_pick").eq(True)).then(
                    pl.col("attack")).alias("radiant_attack"),

                pl.when(pl.col("team").eq(1) & pl.col("is_pick").eq(True)).then(
                    pl.col("attack")).alias("dire_attack"),

                # Hero Roles
                pl.when(pl.col("team").eq(0) & pl.col("is_pick").eq(True)).then(
                    pl.col("roles_vector")).alias("radiant_roles_picks"),

                pl.when(pl.col("team").eq(1) & pl.col("is_pick").eq(True)).then(
                    pl.col("roles_vector")).alias("dire_roles_picks"),

                pl.when(pl.col("team").eq(0) & pl.col("is_pick").eq(False)).then(
                    pl.col("roles_vector")).alias("radiant_roles_bans"),

                pl.when(pl.col("team").eq(1) & pl.col("is_pick").eq(False)).then(
                    pl.col("roles_vector")).alias("dire_roles_bans"),

            ])
            .join(objectives, on="match_id", how="inner")
            .join(exp_adv, on="match_id", how="inner")
            .join(gold_adv, on="match_id", how="inner")
            .filter(
                (pl.col("radiant_picks").list.len() == 5) &
                (pl.col("dire_picks").list.len() == 5) &

                (pl.col("radiant_bans").list.len() == 7) &
                (pl.col("dire_bans").list.len() == 7)
            )
            .with_columns(
                *[
                    pl.col(col).str.to_decimal().alias(col)
                    for col in ["series_type", "series_id", "region"]
                    if isinstance(pl.col(col), pl.String)]
            )
            .with_columns(
                *[pl.col(col).cast(cast, strict=False).alias(col)
                  for col, cast
                  in schema.target_dataset.items()]
            )
            .select(
                *[
                    pl.col({col}).list.drop_nulls()
                    for col in schema.target_dataset.names()
                    if schema.target_dataset[col].is_nested()
                ],
                *[
                    pl.col({col})
                    for col in schema.target_dataset.names()
                    if not schema.target_dataset[col].is_nested()
                ],
            )
        )

        return self.data

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
        )

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
                pl.col("primary_attr")
                .replace(self.dict_attributes,
                         return_dtype=pl.Int32).alias("primary_attribute"),
                pl.col("roles").map_elements(
                    lambda x: [
                        self.dict_roles.get(y) for y in ast.literal_eval(x)]
                    if isinstance(x, str) else x,
                    return_dtype=pl.List(pl.Int32)
                ).replace(self.dict_roles).alias("roles_vector"),
                pl.col("id")
                .cast(pl.Int32)
                .alias("hero_id"),

                pl.col("id")
                .replace(self.dict_hero_index, return_dtype=pl.Int32)
                .alias("hero_idx"),

                pl.col("attack_type")
                .replace(self.dict_attack,
                         return_dtype=pl.Int32).alias("attack"),

                pl.col("localized_name").alias("hero_name"),
            )
        )

        size_roles = len(self.dict_roles)
        _heroes = heroes.collect()
        rows = []
        for h in _heroes.iter_rows(named=True):
            l = len(h["roles_vector"])
            h["roles_vector"] = sorted(
                h["roles_vector"] + [0] * (size_roles - l))
            rows.append(h)
        new_heroes = pl.DataFrame(rows, schema=_heroes.schema)

        return new_heroes.lazy()

    def _items(self) -> pl.LazyFrame:
        items = (
            self._get_lf(ITEMS, cols.items)
            .with_columns(
                pl.col("id").cast(pl.Int32).alias("id"),
                pl.col("id").replace(self.dict_items_index).cast(
                    pl.Int32).alias("item_id"),
                pl.col("name"),
            )
        )
        return items

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
            .with_columns(
                pl.col("hero_id").cast(pl.Int32).alias("hero_id"),
                pl.col("account_id").cast(pl.Int32).alias("account_id"),

                pl.when(pl.col("purchase_gem") == "")
                .then(pl.lit(0))
                .otherwise(pl.col("purchase_gem").str.to_decimal())
                .cast(pl.Int32)
                .alias("purchase_gem"),

                pl.when(pl.col("purchase_rapier") == "")
                .then(pl.lit(0))
                .otherwise(pl.col("purchase_rapier").str.to_decimal())
                .cast(pl.Int32)
                .alias("purchase_rapier"),

                *[pl.col({col}).str
                  .extract_all(r"(\d+)")
                  .list.eval(pl.element().cast(pl.Int32))
                  .alias(f"{col}") for col in cols.players_stats_list],

                *[pl.col(f"item_{x}").replace(self.dict_items_index).alias(f"item_{x}_idx")
                  for x in range(0, 6)],

                *[pl.col(f"backpack_{x}").replace(self.dict_items_index).alias(
                    f"backpack_{x}_idx") for x in range(0, 3)],
            )
            .with_columns(
                pl.concat([
                    pl.col(f"item_{x}_idx")
                    for x in range(0, 6)
                ]).alias("items_vector"),

                pl.concat([
                    pl.col(f"backpack_{x}_idx")
                    for x in range(0, 3)
                ]).alias("backpack_vector"),
            )
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
        lf = lf.match_to_schema(
            schema=schema.metadata_schema, extra_columns='ignore')
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

    def save_dataset(self, year: int) -> pl.DataFrame:
        path = f"{self.data_path}/{year}"
        os.makedirs(path, exist_ok=True)

        df = self.get_year(year).collect()
        ti = df.filter(pl.col("leaguename").str.starts_with(
            f"The International {year}"))
        with open(f"{path}/dataset_schema.txt", "w") as f:
            f.write(str(df.collect_schema()))
        df.write_json(f"{path}/dataset.json")
        ti.write_json(f"{path}/ti.json")

        df = df.sample(fraction=1, seed=42, shuffle=True,
                       with_replacement=True)  # Shuffle the DataFrame
        train_df = df.sample(fraction=0.7, seed=42)
        val_df = df.sample(fraction=0.15, seed=42)
        test_df = df.sample(fraction=0.15, seed=42)

        train_pd = train_df.to_pandas()
        val_pd = val_df.to_pandas()
        test_pd = test_df.to_pandas()

        train_pd.to_json(f"{path}/train.json", orient='records', lines=True)
        val_pd.to_json(f"{path}/val.json", orient='records', lines=True)
        test_pd.to_json(f"{path}/test.json", orient='records', lines=True)

        log.separator()
        log.info(f"Dataset for year {year} saved at {path}/dataset.json")
        log.info(f"Dataset shape: {df.shape}")
        log.info(f"Train set saved at {path}/train.csv")
        log.info(f"Validation set saved at {path}/val.csv")
        log.info(f"Test set saved at {path}/test.csv")
        log.separator()

        return df

    @staticmethod
    def load_json(path: str, year: int) -> pd.DataFrame:
        if not (2021 <= year < 2025):
            raise ValueError("Year must be between 2021 and 2024")
        _path = f"{path}/dota2_data/{year}/dataset.json"
        if not os.path.exists(_path):
            raise FileNotFoundError(
                f"Dataset for year {year} not found at {_path}")
        json = pd.read_json(_path, orient='records', lines=True)
        return json
