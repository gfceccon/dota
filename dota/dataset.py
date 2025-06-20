import ast
import os
import kagglehub
import polars as pl
from .schemas import Schema
from dota.logger import LogLevel, get_logger

log = get_logger(name='Dota Dataset',  log_file='log/dota_dataset.log',)


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

        self.attr_mapping = {
            attr: i + 1 for i, attr
            in enumerate(self.attrs)}

        self.role_mapping = {
            role: i + 1 for i, role
            in enumerate(self.roles)}

        self.hero_mapping = {
            hid: i + 1 for i, hid
            in enumerate(self.heroes_ids)}

        self.item_mapping = {
            hid: i + 1 for i, hid
            in enumerate(self.items_id)}

        self.objetive_type = Schema.objectives_types
        self.objective_team = Schema.objectives_teams
        self.player_team = Schema.players_teams

    def metadata(self) -> pl.LazyFrame:
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
        return pl.concat(scans, how="diagonal_relaxed")

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
        return self.lazy(f"{year}/players.csv")

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
                .cast(pl.Float64, strict=False)
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
                .cast(pl.Float64, strict=False)
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


class Dataset:
    def __init__(self, year: int, path: str = '', log_level=LogLevel.INFO):
        """
        Inicializa o dataset otimizado do Dota 2.
        :param dataset_path: Caminho para o dataset (opcional).
        :param default_year: Ano padrão para carregamento dos dados.
        """
        log.set_level(log_level)
        log.separator()
        log.info("Inicializando Dataset Dota 2...")

        self.config = DatasetHelper(path)
        self.cache_path = None
        self.year = year
        self.check_year()
        self.config.load()
        self.optimizations = pl.QueryOptFlags(
            predicate_pushdown=True,
            projection_pushdown=True,
            simplify_expression=True,
            slice_pushdown=True,
            comm_subplan_elim=True,
            comm_subexpr_elim=True,
            cluster_with_columns=True,
            collapse_joins=True,
            check_order_observe=True,
            fast_projection=True,
        )

    def get(self, path: str = '', slice=False) -> str:
        """
        Carrega e processa os dados do ano especificado, gera o cache ao final e retorna o caminho para o dataset.
        Se o cache já existir, utiliza o cache.
        :param year: Ano dos dados a serem carregados.
        :return: Caminho para o dataset principal.
        """
        self.check_year()
        log.separator()
        cache_dir = os.path.join(self.config.data_path, str(self.year))
        if (path != '' and path is not None and os.path.exists(path)):
            log.info(f"Usando caminho fornecido: {path}")
            cache_dir = path
        data_parquet_path = os.path.join(cache_dir, "data.parquet")
        if os.path.exists(data_parquet_path):
            log.info(
                f"Cache encontrado para o ano {self.year}, carregando do cache...")
            return data_parquet_path
        if (not path):
            log.info(
                f"Cache não encontrado para o ano {self.year}, processando dados...")
        os.makedirs(cache_dir, exist_ok=True)
        log.separator()
        # Carrega e processa os dados principais
        metadata = self.metadata()
        players = self.players()
        heroes = self.config._heroes()
        objectives = self.config._objectives(self.year)
        # Salva arquivos auxiliares
        log.info("Salvando arquivos auxiliares em Parquet...")
        heroes.collect().write_parquet(os.path.join(cache_dir, 'heroes.parquet'))
        objectives.collect().write_parquet(os.path.join(cache_dir, 'objectives.parquet'))
        log.info("Salvando metadata em Parquet...")
        metadata.collect(optimizations=self.optimizations).write_parquet(
            os.path.join(cache_dir, 'metadata.parquet'))
        log.info("Salvando player em Parquet...")
        players.collect(optimizations=self.optimizations).write_parquet(
            os.path.join(cache_dir, 'players.parquet'))
        # Gera o dado principal
        meta_aggregate = [pl.col(col).first().alias(
            col) for col in Schema.metadata_parsed_schema.keys() if col != "match_id"]
        player_aggregate = [pl.col(col).alias(
            col) for col in Schema.players_parsed_schema.keys() if col != "match_id"]
        heroes_aggregate = [pl.col(col).alias(
            col) for col in Schema.heroes_parsed_schema.keys() if col not in ["hero_id"]]
        data = (
            metadata
            .join(players, on="match_id", how="inner")
            .group_by("match_id")
            .agg([
                *player_aggregate,
                *meta_aggregate,
                *heroes_aggregate,
            ])
        )
        log.info("Salvando arquivo principal em Parquet...")
        data = data.collect(optimizations=self.optimizations)
        data.write_parquet(data_parquet_path)
        if slice:
            log.info("Criando um slice temporário do dataset...")
            data = data.sample(n=5)
            data.write_json(os.path.join(cache_dir, 'data.json'))
            log.info("Slice criado com sucesso.")
        log.info(f"Cache salvo em {cache_dir}")
        return data_parquet_path

    def reload_from_cache(self, path: str, slice: bool = False) -> pl.LazyFrame:
        self.check_year()

        cache_dir = os.path.join(self.config.data_path, str(self.year))
        if (path != '' and path is not None and os.path.exists(path)):
            log.info(f"Usando caminho fornecido: {path}")
            cache_dir = path

        if not os.path.exists(cache_dir):
            log.error(
                f"Caminho fornecido {cache_dir} não existe.")
            raise FileNotFoundError(
                f"Caminho fornecido {cache_dir} não existe.")

        data_parquet_path = os.path.join(cache_dir, "data.parquet")
        if os.path.exists(data_parquet_path):
            log.info(
                f"Cache encontrado {data_parquet_path}, sobrescrevendo o cache...")

        if not os.path.join(cache_dir, 'metadata.parquet'):
            log.error('Arquivo metadata.parquet não encontrado')
            raise FileNotFoundError(
                f"Arquivo metadata.parquet não encontrado em {cache_dir}.")
        if not os.path.join(cache_dir, 'players.parquet'):
            log.error('Arquivo players.parquet não encontrado')
            raise FileNotFoundError(
                f"Arquivo players.parquet não encontrado em {cache_dir}.")

        # Carrega e processa os dados principais
        metadata = pl.scan_parquet(os.path.join(cache_dir, 'metadata.parquet'))
        players = pl.scan_parquet(os.path.join(cache_dir, 'players.parquet'))

        # Gera o dado principal
        meta_aggregate = [pl.col(col).first().alias(
            col) for col in Schema.metadata_parsed_schema.keys() if col != "match_id"]
        player_aggregate = [pl.col(col).alias(
            col) for col in Schema.players_parsed_schema.keys() if col != "match_id"]
        heroes_aggregate = [pl.col(col).alias(
            col) for col in Schema.heroes_parsed_schema.keys() if col not in ["hero_id"]]

        data = (
            metadata
            .join(players, on="match_id", how="inner")
            .group_by("match_id")
            .agg([
                *player_aggregate,
                *meta_aggregate,
                *heroes_aggregate,
            ])
        )

        log.info("Salvando arquivo principal em Parquet...")
        data = data.collect()
        data.write_parquet(data_parquet_path)

        if slice:
            log.info("Criando um slice temporário do dataset...")
            data = data.sample(n=5)
            data.write_json(os.path.join(cache_dir, 'data.json'))
            log.info("Slice criado com sucesso.")

        log.info(f"Cache salvo em {cache_dir}")

        return pl.scan_parquet(path)

    def metadata(self) -> pl.LazyFrame:
        """
        Cria e retorna os metadados do dataset, realizando joins e filtros necessários.
        :return: LazyFrame com os metadados processados.
        """
        self.check_year()
        log.separator()
        log.info(f"Criando metadados...")
        min_duration, max_duration = self.config.duration
        metadata = (
            self.config
            .metadata()
            .join(self.config._leagues.lazy(), on="leagueid", how="inner")
            .filter(pl.col("tier").is_in(self.config.tier),
                    pl.col("duration").is_between(min_duration, max_duration))
            .join(self.gold(), on="match_id",)
            .join(self.exp(), on="match_id",)
            .unique(subset=["match_id"])
            .select(Schema.metadata_parsed_schema.keys())
        )
        return metadata

    def players(self) -> pl.LazyFrame:
        """
        Carrega e processa os dados dos jogadores, realizando casts e joins necessários.
        :return: LazyFrame com os dados dos jogadores processados.
        """
        self.check_year()
        log.separator()
        log.info(f"Carregando dados dos jogadores...")
        min_duration, max_duration = self.config.duration
        players = (
            self.config._players(self.year)
            .join(self.config._leagues.lazy(), on="leagueid", how="inner")
            .filter(pl.col("tier").is_in(self.config.tier),
                    pl.col("duration").is_between(min_duration, max_duration))
        )
        # Cast de string para lista nas colunas do schema que são listas
        list_columns = [
            k for k, v in Schema.players_schema.items() if isinstance(v, pl.List)]
        for col in list_columns:
            players = players.with_columns(
                pl.col(col).map_elements(
                    lambda x: ast.literal_eval(x) if isinstance(
                        x, str) and x != "" else x,
                    return_dtype=pl.List(pl.Float64)
                ).alias(col)
            )
        # Cast manual para cada coluna
        players = (
            players
            .with_columns(
                *[pl.col(col).cast(dtype, strict=False).alias(col)
                  for col, dtype in Schema.players_schema.items()],
            )
            .with_columns(
                pl.col("hero_id")
                .cast(pl.Float64, strict=False)
                .replace(self.config.hero_mapping)
                .alias("hero_id"),
            )
        )
        players = self.items_backpack(players)
        players = (
            players
            .join(
                self.config._picks_bans(self.year),
                on=["match_id", "hero_id"],
                how="right",
            )
            .fill_nan(None)
            .fill_null(0)
        )

        select_columns: list[str] = [*[col for col in Schema.players_parsed_schema.keys(
        )], *[col for col in Schema.heroes_parsed_schema.keys() if col not in ["hero_id"]]]

        players = (
            players
            .join(self.config._heroes(), on="hero_id", how="inner")
            .select(select_columns)
        )

        return players

    def gold(self) -> pl.LazyFrame:
        self.check_year()
        log.separator()
        log.info(f"Carregando dados de vantagem de ouro...")
        self.check_year()

        gold_adv = (
            self.config
            ._gold_adv(self.year)
            .sort("minute", descending=False)
            .group_by("match_id", maintain_order=True)
            .agg(pl.col("gold").alias("gold_adv"))
        )

        return gold_adv

    def exp(self) -> pl.LazyFrame:
        self.check_year()
        log.separator()
        log.info(f"Carregando dados de vantagem de experiência...")
        self.check_year()

        exp_adv = (
            self.config
            ._exp_adv(self.year)
            .sort("minute", descending=False)
            .group_by("match_id", maintain_order=True)
            .agg(pl.col("exp").alias("exp_adv"))
        )

        return exp_adv

    def items_backpack(self, data: pl.LazyFrame) -> pl.LazyFrame:
        self.check_year()
        log.separator()
        log.info(f"Processando itens e mochila dos jogadores...")
        data = (
            data.with_columns([
                *[
                    pl.col(f"item_{x}")
                    .replace(self.config.item_mapping)
                    .alias(f"item_{x}_idx")
                    for x in range(0, 6)
                ],

                *[
                    pl.col(f"backpack_{x}")
                    .replace(self.config.item_mapping)
                    .alias(f"backpack_{x}_idx")
                    for x in range(0, 3)],
            ])
            .with_columns([
                pl.concat_list([
                    pl.col(f"item_{x}_idx")
                    for x in range(0, 6)
                ])
                .alias("items_vector"),

                pl.concat_list([
                    pl.col(f"backpack_{x}_idx")
                    for x in range(0, 3)
                ])
                .alias("backpack_vector"),
            ])
            .drop([
                *[
                    f"item_{x}"
                    for x in range(0, 6)
                ],
                *[
                    f"item_{x}_idx"
                    for x in range(0, 6)
                ],
                *[
                    f"backpack_{x}"
                    for x in range(0, 3)
                ],
                *[
                    f"backpack_{x}_idx"
                    for x in range(0, 3)
                ]])
        )
        return data

    def check_year(self) -> None:
        self.config.check_year(self.year)

    @staticmethod
    def force_download() -> str:
        """
        Força o download do dataset completo do KaggleHub.
        :param path: Caminho onde o dataset será salvo (opcional).
        :return: Caminho do dataset baixado.
        """
        log.separator()
        log.info("Forçando download do dataset completo...")
        return kagglehub.dataset_download(handle=DATASET, force_download=True,)

    @staticmethod
    def save_year(year: int, path: str = '') -> str:
        """
        Salva o dataset do ano especificado em um arquivo CSV.
        :param year: Ano dos dados a serem salvos.
        :param path: Caminho onde o dataset será salvo (opcional).
        :return: Caminho do dataset salvo.
        """
        log.separator()
        log.info(f"Salvando dataset do ano {year}...")
        ds = Dataset(year, path)
        _path = ds.get(path)
        log.info(f"Dataset salvo em {_path}")
        return _path

    def reset_cache(self, slice: bool = True) -> None:
        """
        Remove o cache do dataset para o ano especificado.
        :return: None
        """
        log.separator()
        log.info(f"Removendo cache do dataset para o ano {self.year}...")
        cache_dir = os.path.join(self.config.data_path, str(self.year))
        if os.path.exists(cache_dir):
            for file in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            log.info(f"Cache removido em {cache_dir}")
        else:
            log.warn(f"Nenhum cache encontrado para o ano {self.year}.")

        self.get(slice=slice)
