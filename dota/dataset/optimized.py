import ast
import os
import kagglehub
import polars as pl
from dota.logger import get_logger
from .optimized_schema import OptimizedSchema
import pandas as pd

log = get_logger()


class OptimizedConfig:
    dataset_name = 'bwandowando/dota-2-pro-league-matches-2023'
    data_path = '/tmp/dota/dataset/'
    LEAGUES = f"Constants/Constants.Leagues.csv"
    HEROES = f"Constants/Constants.Heroes.csv"
    ITEMS = f"Constants/Constants.Items.csv"
    PATCHES = f"Constants/Constants.Patch.csv"

    def METADATA(self, x): return f"{x}/main_metadata.csv"
    def OBJECTIVES(self, x): return f"{x}/objectives.csv"
    def PICKS_BANS(self, x): return f"{x}/picks_bans.csv"
    def PLAYERS(self, x): return f"{x}/players.csv"
    def EXP_ADV(self, x): return f"{x}/radiant_exp_adv.csv"
    def GOLD_ADV(self, x): return f"{x}/radiant_gold_adv.csv"
    def TEAM_FIGHTS(self, x): return f"{x}/teamfights.csv"

    _leagues: pl.DataFrame
    _heroes: pl.DataFrame
    _items: pl.DataFrame
    _patches: pl.DataFrame

    def __init__(self,
                 path: str = '',
                 duration: tuple[int, int] = (10 * 60, 180 * 60),
                 tier: list[str] = ['professional', 'premium'],
                 years: tuple[int, int] = (2021, 2024),):
        super().__init__()
        log.separator()
        self.dataset_path = self.try_load_cache(path)
        log.info("Initializing dataset configuration...")
        log.info(f"Dataset name: {self.dataset_name}")
        log.info(f"Dataset path: {self.dataset_path}")
        log.info(f"Data path: {self.data_path}")

        os.makedirs(self.data_path, exist_ok=True)
        self.start_year, self.end_year = years
        self.duration = duration
        self.tier = tier

    def load(self) -> None:
        self.load_constants()
        self.load_constants_ids()
        self.load_mappings()

    def load_constants(self):
        log.separator()
        log.info("Loading constants...")
        self._leagues = pl.read_csv(
            os.path.join(self.dataset_path, self.LEAGUES))
        self._heroes = pl.read_csv(
            os.path.join(self.dataset_path, self.HEROES))
        self._items = pl.read_csv(os.path.join(self.dataset_path, self.ITEMS))
        self._patches = pl.read_csv(
            os.path.join(self.dataset_path, self.PATCHES))
        log.info(f"Leagues: {self._leagues.shape}")
        log.info(f"Heroes: {self._heroes.shape}")
        log.info(f"Items: {self._items.shape}")
        log.info(f"Patches: {self._patches.shape}")

    def load_constants_ids(self) -> None:
        log.separator()
        log.info("Loading constants IDs...")

        self.attrs = (
            self._heroes.select(pl.col("primary_attr"))
            .unique().to_dict(as_series=False)["primary_attr"]
        )

        self.roles: list[str] = list({
            role for roles_list in
            self._heroes.select("roles")
            .to_series().to_list()
            for role
            in (ast.literal_eval(roles_list)
                if isinstance(roles_list, str) else roles_list)
        })

        self.heroes_ids = (
            self._heroes.select(pl.col("id"))
            .unique().sort("id")
            .to_series().to_list())

        self.items_id = (
            self._items.select(pl.col("id"))
            .unique().sort("id")
            .to_series().to_list())

    def load_mappings(self) -> None:
        log.separator()
        log.info("Loading IDs mappings...")

        self.attack_mapping = {
            "Melee": 1,
            "Ranged": 2,
            "Unknown": 0
        }

        self.attr_mapping = {
            attr: i + 1 for i, attr
            in enumerate(self.attrs)}

        self.role_mapping = {
            role: i + 1 for i, role
            in enumerate(self.roles)}

        self.hero_mapping = {
            hid: i for i, hid
            in enumerate(self.heroes_ids)}

        self.item_mapping = {
            hid: i + 1 for i, hid
            in enumerate(self.items_id)}

        self.objetive_type = OptimizedSchema.objectives_types
        self.objective_team = OptimizedSchema.objectives_teams
        self.player_team = OptimizedSchema.players_teams

    def metadata(self) -> pl.LazyFrame:
        log.separator()
        log.info(
            f"Loading metadata from {self.start_year} to {self.end_year}...")

        scans: list[pl.LazyFrame] = []
        schemas: list[pl.Schema] = []
        files = [
            self.METADATA(year) for year
            in range(self.start_year, self.end_year + 1)]

        for file in files:
            _lf = pl.scan_csv(f"{self.dataset_path}/{file}")
            scans.append(_lf)
            schemas.append(_lf.collect_schema())

        names = set(schemas[0].names())
        lost = []

        for schema in schemas:
            names.intersection_update(schema.names())
            lost.extend(set(schema.names()) - names)

        lf = pl.concat(scans, how="diagonal_relaxed").lazy()
        log.warn(f"Columns lost during metadata loading: {lost}")
        return lf

    def lazy(self, dataset: str = '') -> pl.LazyFrame:
        log.separator()
        log.info(f"Loading {dataset} as LazyFrame...")
        if not os.path.exists(os.path.join(self.dataset_path, dataset)):
            log.error(
                f"Dataset file {dataset} does not exist in {self.dataset_path}.")
            raise FileNotFoundError(
                f"Dataset file {dataset} does not exist in {self.dataset_path}.")
        return pl.scan_csv(os.path.join(self.dataset_path, dataset))

    def try_load_cache(self, path: str) -> str:
        if (path == '' or path is None):
            path = os.path.expanduser(
                f'~/.cache/kagglehub/datasets/{self.dataset_name}/versions/')
            if (os.path.exists(path)):
                log.info(f"Checking cache path: {path}")
                paths = os.listdir(path)
                if (len(paths) > 0):
                    path = os.path.join(path, paths[0])
                    log.info(
                        f"Dataset path found in cache: {path}")
                else:
                    log.error(
                        "Dataset path is not provided and cannot be found in KaggleHub.")
                    path = ''
        if (path == '' or path is None):
            path = kagglehub.dataset_download(handle=self.dataset_name,)
        return path

    def _players(self, year: int) -> pl.LazyFrame:
        if not (self.start_year <= year <= self.end_year):
            log.error(
                f"Year {year} is out of range ({self.start_year}-{self.end_year}).")
            raise ValueError(
                f"Year {year} is out of range ({self.start_year}-{self.end_year}).")
        return self.lazy(self.PLAYERS(year))
    
    def _objectives(self, year: int) -> pl.LazyFrame:
        if not (self.start_year <= year <= self.end_year):
            log.error(
                f"Year {year} is out of range ({self.start_year}-{self.end_year}).")
            raise ValueError(
                f"Year {year} is out of range ({self.start_year}-{self.end_year}).")
        return self.lazy(self.OBJECTIVES(year))
    
    def _picks_bans(self, year: int) -> pl.LazyFrame:
        if not (self.start_year <= year <= self.end_year):
            log.error(
                f"Year {year} is out of range ({self.start_year}-{self.end_year}).")
            raise ValueError(
                f"Year {year} is out of range ({self.start_year}-{self.end_year}).")
        return self.lazy(self.OBJECTIVES(year))
    
    def _exp_adv(self, year: int) -> pl.LazyFrame:
        if not (self.start_year <= year <= self.end_year):
            log.error(
                f"Year {year} is out of range ({self.start_year}-{self.end_year}).")
            raise ValueError(
                f"Year {year} is out of range ({self.start_year}-{self.end_year}).")
        return self.lazy(self.EXP_ADV(year))
    
    def _gold_adv(self, year: int) -> pl.LazyFrame:
        if not (self.start_year <= year <= self.end_year):
            log.error(
                f"Year {year} is out of range ({self.start_year}-{self.end_year}).")
            raise ValueError(
                f"Year {year} is out of range ({self.start_year}-{self.end_year}).")
        return self.lazy(self.GOLD_ADV(year))
    
    def _team_fights(self, year: int) -> pl.LazyFrame:
        if not (self.start_year <= year <= self.end_year):
            log.error(
                f"Year {year} is out of range ({self.start_year}-{self.end_year}).")
            raise ValueError(
                f"Year {year} is out of range ({self.start_year}-{self.end_year}).")
        return self.lazy(self.TEAM_FIGHTS(year))

class OptimizedDataset:
    def __init__(self, dataset_path: str = ''):
        log.separator()
        log.info("Initializing Dota 2 Dataset...")

        self.config = OptimizedConfig(dataset_path)
        self.config.load()

    def get(self, year: int) -> pl.LazyFrame:
        if not (self.config.start_year <= year <= self.config.end_year):
            log.error(
                f"Year {year} is out of range ({self.config.start_year}-{self.config.end_year}).")
            raise ValueError(
                f"Year {year} is out of range ({self.config.start_year}-{self.config.end_year}).")

        metadata = self.config.metadata()
        return metadata

    def players(self, year: int) -> pl.LazyFrame:
        log.separator()
        log.info(f"Loading players data for year {year}...")

        if not (self.config.start_year <= year <= self.config.end_year):
            log.error(
                f"Year {year} is out of range ({self.config.start_year}-{self.config.end_year}).")
            raise ValueError(
                f"Year {year} is out of range ({self.config.start_year}-{self.config.end_year}).")

        players = self.config._players(year)
        
        # Cast de string para lista nas colunas do schema que são listas
        players_schema_dict = OptimizedSchema.players_schema_dict
        list_columns = [k for k, v in players_schema_dict.items() if isinstance(v, pl.List)]

        for col in list_columns:
            players = players.with_columns(
                pl.col(col).map_elements(lambda x: ast.literal_eval(x) if isinstance(x, str) and x != "" else x, return_dtype=pl.List(pl.Float64)).alias(col)
            )

        # Cast manual para cada coluna
        players = (
            players.with_columns([
            pl.col(col).cast(dtype, strict=False).alias(col)
            for col, dtype in players_schema_dict.items()
        ])
        .select(
            [pl.col(col) for col in players_schema_dict.keys()]
        ))
        return players