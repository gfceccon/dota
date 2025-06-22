import ast
import os
import kagglehub
import polars as pl
from .schemas import Schema
from dota.logger import get_logger

log = get_logger(name='Dota Dataset Helper',
                 log_file='log/dota_dataset_helper.log',)


DATASET = 'bwandowando/dota-2-pro-league-matches-2023'


class DatasetHelper:
    def __init__(self,
                 path: str = '',
                 duration: tuple[int, int] = (10 * 60, 180 * 60),
                 tier: list[str] = ['professional', 'premium'],
                 years: tuple[int, int] = (2021, 2024),):

        self.dataset_name = DATASET
        self.data_path = 'tmp/dota/dataset/'

        log.separator()
        log.info("Inicializando configuração do dataset...")
        os.makedirs(self.data_path, exist_ok=True)
        self.dataset_path = self.try_load_cache(path)

        self._leagues: pl.DataFrame
        self._heroes_data: pl.DataFrame
        self._items: pl.DataFrame
        self._patches: pl.DataFrame
        self.start_year, self.end_year = years
        self.duration = duration
        self.tier = tier

        log.info(f"Nome do dataset: {self.dataset_name}")
        log.info(f"Caminho do dataset: {self.dataset_path}")
        log.info(f"Caminho dos dados: {self.data_path}")

    def load(self) -> None:
        self.load_constants()
        self.load_constants_ids()
        self.load_mappings()

    def load_constants(self):
        log.separator()
        log.info("Carregando constantes...")
        self._leagues = pl.read_csv(
            os.path.join(self.dataset_path, "Constants/Constants.Leagues.csv"))
        self._heroes_data = pl.read_csv(
            os.path.join(self.dataset_path, "Constants/Constants.Heroes.csv"))
        self._items = pl.read_csv(
            os.path.join(self.dataset_path, "Constants/Constants.Items.csv"))
        self._patches = (
            pl.
            read_csv(
                os.path.join(self.dataset_path, "Constants/Constants.Patch.csv"))
            .with_columns(
                pl.col("patch").alias("patch_version"),
                pl.col("date").cast(pl.Datetime).alias("patch_date"),)
            .drop("patch", "date")
        )

        log.info(f"Ligas: {self._leagues.shape}")
        log.info(f"Heróis: {self._heroes_data.shape}")
        log.info(f"Itens: {self._items.shape}")
        log.info(f"Patches: {self._patches.shape}")

    def load_constants_ids(self) -> None:
        log.separator()
        log.info("Carregando IDs das constantes...")

        self.attrs = (
            self._heroes_data.select(pl.col("primary_attr"))
            .unique().to_dict(as_series=False)["primary_attr"])

        self.roles: list[str] = list({
            role for roles_list in
            self._heroes_data.select("roles")
            .to_series().to_list()
            for role
            in (ast.literal_eval(roles_list)
                if isinstance(roles_list, str) else roles_list)
        })

        self.heroes_ids = (
            self._heroes_data.select(pl.col("id"))
            .unique().sort("id")
            .to_series().to_list())

        self.items_id = (
            self._items.select(pl.col("id"))
            .unique().sort("id")
            .to_series().to_list())

    def load_mappings(self) -> None:
        log.separator()
        log.info("Carregando mapeamentos de IDs...")

        self.attack_mapping = {
            "Melee": 1,
            "Ranged": 2,
            "Unknown": 0
        }

        self.attr_mapping: dict[int, int] = {
            attr: i + 1 for i, attr
            in enumerate(self.attrs)}

        self.role_mapping: dict[str, int] = {
            role: i + 1 for i, role
            in enumerate(self.roles)}

        self.hero_mapping: dict[int, int] = {
            hid: i + 1 for i, hid
            in enumerate(self.heroes_ids)}

        self.item_mapping: dict[int, int] = {
            hid: i + 1 for i, hid
            in enumerate(self.items_id)}

        self.objetive_type = Schema.objectives_types
        self.objective_team = Schema.objectives_teams
        self.player_team = Schema.players_teams

    def _metadata(self, year: int) -> pl.LazyFrame:
        self.check_year(year)
        return self.lazy(f"{year}/main_metadata.csv")

    def all_metadata(self, year: int) -> pl.LazyFrame:
        log.separator()
        log.info(
            f"Carregando metadados de {self.start_year} até {self.end_year}...")

        scans: list[pl.LazyFrame] = []
        schemas: list[pl.Schema] = []
        files = [f"{year}/main_metadata.csv" for year
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

        log.warn(
            f"Colunas perdidas durante o carregamento dos metadados: {set(lost)}")

        concat = pl.concat(scans, how="diagonal_relaxed")
        log.info(f"Total de arquivos carregados: {len(scans)}")
        log.info(f"Total de colunas: {len(names)}")
        log.info(f"Colunas encontradas: {names}")
        return concat.select(*names)

    def all_players(self) -> pl.LazyFrame:
        log.separator()
        log.info(
            f"Carregando players de {self.start_year} até {self.end_year}...")

        scans: list[pl.LazyFrame] = []
        schemas: list[pl.Schema] = []
        files = [f"{year}/players.csv" for year
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

        log.warn(
            f"Colunas perdidas durante o carregamento dos metadados: {set(lost)}")

        concat = pl.concat(scans, how="diagonal_relaxed")
        log.info(f"Total de arquivos carregados: {len(scans)}")
        log.info(f"Total de colunas: {len(names)}")
        log.info(f"Colunas encontradas: {names}")
        return concat.select(*names)

    def lazy(self, dataset: str = '') -> pl.LazyFrame:
        log.separator()
        log.info(f"Carregando {os.path.join(self.dataset_path, dataset)}...")
        if not os.path.exists(os.path.join(self.dataset_path, dataset)):
            log.error(
                f"Arquivo do dataset {dataset} não existe em {self.dataset_path}.")
            raise FileNotFoundError(
                f"Arquivo do dataset {dataset} não existe em {self.dataset_path}.")
        return pl.scan_csv(os.path.join(self.dataset_path, dataset))

    def try_load_cache(self, path: str) -> str:
        if (path == '' or path is None):
            path = os.path.expanduser(
                f'~/.cache/kagglehub/datasets/{self.dataset_name}/versions/')
            if (os.path.exists(path)):
                log.info(f"Verificando caminho do cache: {path}")
                paths = os.listdir(path)
                if (len(paths) > 0):
                    latest = max(paths)
                    path = os.path.join(path, latest)
                    log.info(
                        f"Caminho do dataset encontrado no cache: {path}")
                else:
                    log.error(
                        "Caminho do dataset não foi fornecido e não pode ser encontrado no KaggleHub.")
                    path = ''
        if (path == '' or path is None):
            path = kagglehub.dataset_download(handle=self.dataset_name,)
        return path

    def _players(self, year: int) -> pl.LazyFrame:
        self.check_year(year)
        list_columns = [
            k for k, v in Schema.players_schema.items() if isinstance(v, pl.List)]
        return (
            self.lazy(f"{year}/players.csv")
            .with_columns(
                pl.col("hero_id")
                .cast(pl.Int64, strict=False)
                .replace(self.hero_mapping)
                .alias("hero_id"),
            )
            .with_columns(
                pl.col(col).fill_null("[]").map_elements(
                    lambda x: ast.literal_eval(x if x != "" else "[]") if isinstance(
                        x, str) and x != "" else x,
                    return_dtype=pl.List(pl.Float64)
                ).alias(col)
                for col in list_columns
            )
            .with_columns(
                *[pl.col(col).cast(dtype, strict=False).fill_null(strategy='zero').alias(col)
                  for col, dtype in Schema.players_schema.items()],
            )
        )

    def _objectives(self, year: int) -> pl.LazyFrame:
        self.check_year(year)
        return (
            self.lazy(f"{year}/objectives.csv")
            .with_columns(
                pl.col("type")
                .replace(self.objetive_type)
                .alias("type"),
                pl.when(pl.col("team") != "")
                .then(
                    pl.col("team")
                    .replace(self.objective_team))
                .alias("team")
            ))

    def _picks_bans(self, year: int) -> pl.LazyFrame:
        self.check_year(year)
        return (
            self.lazy(f"{year}/picks_bans.csv")
            .with_columns(
                pl.col("hero_id")
                .cast(pl.Int64, strict=False)
                .replace(self.hero_mapping)
                .alias("hero_id"),
            ))

    def _exp_adv(self, year: int) -> pl.LazyFrame:
        self.check_year(year)
        return self.lazy(f"{year}/radiant_exp_adv.csv")

    def _gold_adv(self, year: int) -> pl.LazyFrame:
        self.check_year(year)
        return self.lazy(f"{year}/radiant_gold_adv.csv")

    def _team_fights(self, year: int) -> pl.LazyFrame:
        self.check_year(year)
        return self.lazy(f"{year}/team_fights.csv")

    def _heroes(self) -> pl.LazyFrame:
        log.separator()
        log.info("Carregando dados dos heróis...")
        heroes = (
            self._heroes_data
            .with_columns(
                pl.col("roles")
                .map_elements(
                    lambda x: ast.literal_eval(x)
                    if isinstance(x, str) and x != "" else x,
                    return_dtype=pl.List(pl.String))
                .alias("roles")
            )
            .with_columns([
                pl.col("id")
                .replace(self.hero_mapping)
                .cast(pl.Int64, strict=False)
                .alias("hero_id"),

                pl.col("localized_name")
                .alias("hero_name"),

                pl.col("primary_attr")
                .replace(self.attr_mapping)
                .cast(pl.Int64)
                .alias("primary_attribute"),

                pl.col("roles").list
                .eval(
                    pl.element()
                    .replace(self.role_mapping)
                )
                .cast(pl.List(pl.Int64))
                .alias("roles_vector"),

                pl.col("attack_type")
                .replace(self.attack_mapping)
                .cast(pl.Int64)
                .alias("attack_type"),

                *[pl.col(k).cast(v, strict=False).alias(k)
                  for k, v in Schema.heroes_schema.items()]
            ])
        )
        size = len(self.role_mapping)
        rows = []
        for h in heroes.iter_rows(named=True):
            del h[""]
            del h["id"]
            del h["localized_name"]
            del h["roles"]
            l = len(h["roles_vector"])
            h["roles_vector"] = sorted(h["roles_vector"] + [0] * (size - l))
            rows.append(h)
        return pl.LazyFrame(rows)

    def check_year(self, year: int) -> None:
        if not (self.start_year <= year <= self.end_year):
            log.error(
                f"Ano {year} está fora do intervalo ({self.start_year}-{self.end_year}).")
            raise ValueError(
                f"Ano {year} está fora do intervalo ({self.start_year}-{self.end_year}).")
