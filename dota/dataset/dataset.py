import ast
from enum import Enum
from typing import Optional
import polars as pl

from dota.logger import LogLevel, get_logger
from . import PATH

log = get_logger('dataset', LogLevel.INFO, False)


def METADATA(x): return f"{x}/main_metadata.csv"
def DRAFT(x): return f"{x}/draft_timings.csv"
def OBJECTIVES(x): return f"{x}/objectives.csv"
def PICKS_BANS(x): return f"{x}/picks_bans.csv"
def PLAYERS(x): return f"{x}/players.csv"
def EXP_ADV(x): return f"{x}/radiant_exp_adv.csv"
def GOLD_ADV(x): return f"{x}/radiant_gold_adv.csv"
def TEAM_FIGHTS(x): return f"{x}/teamfights.csv"


LEAGUES = f"Constants/Constants.Leagues.csv"
HEROES = f"Constants/Constants.Heroes.csv"
PATCHES = f"Constants/Constants.Patch.csv"


class Dataset:

    def __init__(self,
                 duration: tuple[int, int] = (30 * 60, 120 * 60),
                 tier: list[str] = ['professional'],
                 years: tuple[int, int] = (2020, 2025)):
        self.years = years
        self.duration = duration
        self.tier = tier

        _all_heroes = self._get_lf(HEROES, 'hero')
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

        self.heroes = self._heroes()
        self.patches = self._patches()
        self.metadata = self._metadata()
        self.leagues = self._leagues()

    def get_heroes_usage(self) -> pl.LazyFrame:
        heroes = self._heroes()
        for year in range(self.years[0], self.years[1]):
            lf = self.get_year(year)
            heroes = (
                lf.select("match_id", "radiant_picks",
                          "dire_picks", "radiant_bans", "dire_bans")
                .group_by("match_id")
                .agg([
                    pl.col("radiant_picks").list.eval(pl.element().alias(
                        "hero")).explode().alias("radiant_picks"),
                    pl.col("dire_picks").list.eval(pl.element().alias(
                        "hero")).explode().alias("dire_picks"),
                    pl.col("radiant_bans").list.eval(pl.element().alias(
                        "hero")).explode().alias("radiant_bans"),
                    pl.col("dire_bans").list.eval(pl.element().alias(
                        "hero")).explode().alias("dire_bans")
                ])
            )

        return heroes

    def get_year(self, year: int) -> pl.LazyFrame:
        if year < self.years[0] or year >= self.years[1]:
            raise ValueError(
                f"Year {year} is out of range {self.years[0]}-{self.years[1]}")
        metadata = self._get_lf(METADATA(str(year)), 'metadata')
        files = [
            self._get_lf(DRAFT(str(year)),  'draft'),
            self._get_lf(OBJECTIVES(str(year)), 'objectives'),
            self._get_lf(PICKS_BANS(str(year)),  'picks_bans'),
            self._get_lf(PLAYERS(str(year)),  'players'),
            self._get_lf(EXP_ADV(str(year)), 'exp_adv'),
            self._get_lf(GOLD_ADV(str(year)),  'gold_adv'),
            self._get_lf(TEAM_FIGHTS(str(year)), 'team_fights'),
        ]
        for data in files:
            metadata = metadata.join(data, left_on="match_id", how="left")

        metadata = (
            metadata
            .join(self.leagues, on="league_id", how="left")
            .join(self.patches, on="patch", how="left")
            .join(self.heroes, on="hero_id", how="left")
        )
        return self._preprocess(metadata)

    def _preprocess(self, data: pl.LazyFrame) -> pl.LazyFrame:
        data = (
            data
            .drop_nulls("match_id")
            .filter(
                (pl.col("duration").is_between(self.duration[0], self.duration[1])) &
                (pl.col("league_tier").is_in(self.tier)) &
                (pl.col("match_id").is_not_null())
            )
            .group_by("match_id")
            .agg([
                # Hero picks and bans
                pl.when(pl.col("team").eq(0) & pl.col("pick").eq(True)).then(
                    pl.col("hero_idx") + 1).drop_nulls().alias("radiant_picks_idx"),
                pl.when(pl.col("team").eq(1) & pl.col("pick").eq(True)).then(
                    pl.col("hero_idx") + 1).drop_nulls().alias("dire_picks_idx"),

                pl.when(pl.col("team").eq(0) & pl.col("pick").eq(False)).then(
                    pl.col("hero_idx") + 1).drop_nulls().alias("radiant_bans_idx"),
                pl.when(pl.col("team").eq(1) & pl.col("pick").eq(False)).then(
                    pl.col("hero_idx") + 1).drop_nulls().alias("dire_bans_idx"),

                # Split players

            ])
        )
        return data

    def _patches(self) -> pl.LazyFrame:
        patch_count: dict[int, tuple[int, str]] = {}
        patches_detail = self._get_lf(PATCHES, "patch").collect()
        meta = self._metadata().drop_nans("match_id")
        patch_meta = (
            meta.select("match_id", "patch")
            .group_by("patch")
            .agg(pl.count("match_id").alias("count"), pl.col("start_date_time").cast(pl.Datetime).dt.year().unique().alias("year"))
            .select("patch", "count", "year").collect()
        )
        for row in patch_meta.iter_rows(named=True):
            patch = row["patch"]
            count = row["count"]
            patch_name = patches_detail.item(patch - 1, 0)
            patch_count[patch] = (count, patch_name)
        return pl.LazyFrame(
            {
                "patch": list(patch_count.keys()),
                "count": [x[0] for x in patch_count.values()],
                "name": [x[1] for x in patch_count.values()]
            }
        ).sort("patch")

    def _leagues(self) -> pl.LazyFrame:
        leagues = (
            self._get_lf(LEAGUES, "league")
            .filter(pl.col("tier").is_in(self.tier))
            .select([
                pl.col("leagueid").alias("league_id"),
                pl.col("leaguename").alias("league_name"),
                pl.col("tier").alias("league_tier")
            ])
        )
        return leagues

    def _heroes(self, ):
        heroes = (
            self._get_lf(HEROES, "hero")
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

                pl.col("id").map_elements(lambda x: self.dict_hero_index.get(
                    x), return_dtype=pl.Int32).alias("hero_idx"),
            )
            .with_columns(
                pl.col("roles").map_elements(
                    lambda x: [1 if i in x else 0 for i in self.roles_idx],
                    return_dtype=pl.List(pl.Int32)
                ).alias("roles_vector"),
                pl.col("id").alias("hero_id").cast(pl.Int32),
                pl.col("localized_name").alias("hero_name"),
            )
        )
        return heroes

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
            _lf = pl.scan_csv(file)
            scans.append(_lf)
            schemas.append(_lf.collect_schema())
        names = set(schemas[0].names())
        lost = []

        for schema in schemas:
            names.intersection_update(schema.names())
            lost.extend(set(schema.names()) - names)
        lf = pl.concat(scans, how="diagonal_relaxed").lazy()
        return lf, lost

    def _get_lf(self, file: str, prefix: str | None = None) -> pl.LazyFrame:
        lf = pl.scan_csv(f"{PATH}/{file}")
        if(prefix is None):
            return lf
        lf = (
            lf.with_columns(
                [pl.col(_col).alias(f"{prefix}_{_col}")
                 for _col in lf.columns])
            .with_columns(
                pl.col(f"{prefix}_match_id").cast(pl.Int64).alias("match_id"),
                pl.col(f"{prefix}_team").cast(pl.UInt8).alias("team"),
                pl.col(f"{prefix}_hero_id").cast(pl.Int32).alias("hero_id"),
                pl.col(f"{prefix}_is_pick").cast(pl.Boolean).alias("pick"),
                pl.col(f"{prefix}_order").cast(pl.UInt8).alias("order"),
                pl.col(f"{prefix}_patch").cast(pl.Int32).alias("patch"),
                pl.col(f"{prefix}_leagueid").cast(pl.Int32).alias("league_id"),
            )
        )
        return lf
