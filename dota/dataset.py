import ast
import os
import kagglehub
import polars as pl
from .schemas import Schema
from dota.logger import LogLevel, get_logger
from dota.dataset_helper import DatasetHelper

log = get_logger(name='Dota Dataset',  log_file='log/dota_dataset.log',)


DATASET = 'bwandowando/dota-2-pro-league-matches-2023'
CAPTAINS_MODE = 2


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
        players, players_col = self.players()
        heroes = self.config._heroes()
        objectives = self.config._objectives(self.year)
        log.separator()
        log.info("Salvando arquivos auxiliares...")
        heroes.collect().write_parquet(os.path.join(cache_dir, 'heroes.parquet'))
        objectives.collect().write_parquet(os.path.join(cache_dir, 'objectives.parquet'))

        log.info("Salvando metadata...")
        metadata.collect(optimizations=self.optimizations).write_parquet(
            os.path.join(cache_dir, 'metadata.parquet'))

        log.info("Salvando player...")
        players.collect(optimizations=self.optimizations).write_parquet(
            os.path.join(cache_dir, 'players.parquet'))

        # Gera o dado principal
        meta_aggregate = [pl.col(col).first().alias(
            col) for col in Schema.metadata_parsed_schema.keys() if col != "match_id"]

        player_aggregate = [pl.col(col).alias(
            col) for col in players_col if col not in ["match_id"]]

        data = (
            metadata
            .join(players, on="match_id", how="inner")
            .group_by("match_id")
            .agg([
                *meta_aggregate,
                *player_aggregate,

            ])
        )
        log.info("Salvando arquivo principal...")
        data = data.collect(optimizations=self.optimizations)
        data.write_parquet(data_parquet_path)
        if slice:
            log.info("Criando um slice temporário do dataset...")
            data = data.sample(n=1)
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
        meta_aggregate = [pl.col(col).drop_nulls().first().alias(
            col) for col in Schema.metadata_parsed_schema.keys() if col != "match_id"]
        player_aggregate = [pl.col(col).drop_nulls().alias(
            col) for col in Schema.players_parsed_schema.keys() if col != "match_id"]
        heroes_aggregate = [pl.col(col).drop_nulls().alias(
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
        metadata = self.filter((
            self.config._metadata(self.year)
            .unique(subset=["match_id"])
            .join(self.config._leagues.lazy(), on="leagueid", how="inner")
        ))
        metadata = (
            metadata
            .join(self.gold(), on="match_id",)
            .join(self.exp(), on="match_id",)
            .select(Schema.metadata_parsed_schema.keys())
        )
        return metadata

    def both(self) -> pl.LazyFrame:
        self.check_year()
        log.separator()
        log.info(f"Carregando dados principais...")
        heroes_lf = self.config._heroes()
        players_lf = self.config._players(self.year)
        picks_bans_lf = self.config._picks_bans(self.year)
        leagues_lf = self.config._leagues.lazy()
        metadata = self.config._metadata(self.year)
        min_duration, max_duration = self.config.duration
        tiers = self.config.tier

        select_columns: list[str] = [
            *[col for col in Schema.players_parsed_schema.keys()
                if col not in Schema.players_parsed_normalize.keys()],

            *[col for col in Schema.heroes_parsed_schema.keys()
                if col not in Schema.heroes_parsed_normalize.keys()],
        ]

        select_columns_set = set(select_columns)
        select_columns = list(select_columns_set)


        meta_aggregate = [
            pl.col(col).first().alias(col)
            for col in Schema.metadata_parsed_schema.keys() if col != "match_id"]

        player_aggregate = [
            pl.col(col).alias(col)
            for col in Schema.players_parsed_schema.keys() if col not in ["match_id"]]

        data = (
            metadata
            .unique(subset=["match_id"])
            .join(leagues_lf, on="leagueid", suffix="_league")
            .filter(
                pl.col("match_id").is_not_null(),
                pl.col("duration").is_between(min_duration, max_duration),
                pl.col("radiant_win").is_not_null(),
                pl.col("leagueid").is_not_null(),
                pl.col("game_mode").eq(CAPTAINS_MODE),
                pl.col("tier").is_in(tiers),
            )
            .join(picks_bans_lf, on="match_id", suffix="_picks_bans")
            .join(players_lf, on=["match_id", "hero_id"], how="left", suffix="_players")
            .join(self.gold(), on="match_id", suffix="_gold")
            .join(self.exp(), on="match_id", suffix="_exp")
            .join(heroes_lf, on="hero_id")
            .with_columns([
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
            .group_by("match_id")
            .agg([
                *meta_aggregate,
                *player_aggregate,
            ])
        )
        log.separator()
        log.info("Carregando dados principais...")
        data = data.collect(optimizations=self.optimizations)
        log.info(f"Dados principais carregados com sucesso.")
        log.separator()
        log.info("Criando um slice temporário do dataset...")
        slice = data.sample(1)
        slice.write_json('data.json')

        return data.lazy()

    def players(self) -> tuple[pl.LazyFrame, list[str]]:
        """
        Carrega e processa os dados dos jogadores, realizando casts e joins necessários.
        :return: LazyFrame com os dados dos jogadores processados.
        """
        self.check_year()
        log.separator()
        log.info(f"Carregando dados dos jogadores...")
        heroes_lf = self.config._heroes()
        players_lf = self.config._players(self.year)
        picks_bans_lf = self.config._picks_bans(self.year)
        leagues_lf = self.config._leagues.lazy()
        log.separator()

        players_league = self.filter(
            players_lf
            .join(leagues_lf, on="leagueid", how="inner")
        )

        players_items = self.items_backpack(players_league)

        players_picks_bans = (
            players_items
            .join(picks_bans_lf, on=["match_id", "hero_id"], how="right")
        )

        players = (
            players_picks_bans
            .join(heroes_lf, on="hero_id", how="inner")
        )

        select_columns: list[str] = [
            *[
                col
                for col
                in Schema.players_parsed_schema.keys()
                if col not in Schema.players_parsed_normalize.keys()
            ],
            *[
                col
                for col
                in Schema.heroes_parsed_schema.keys()
                if col not in Schema.heroes_parsed_normalize.keys()
            ],
        ]
        select_columns_set = set(select_columns)  # Remove duplicates
        select_columns = list(select_columns_set)  # Convert back to list

        log.separator()
        return players.select(select_columns), select_columns

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

    def robust_normalize(self, df: pl.LazyFrame, cols: list, group_keys: list) -> pl.LazyFrame:
        """
        Aplica normalização robusta (Robust Scaler) para as colunas especificadas, agrupando por group_keys.
        Cria colunas _radiant e _dire para cada feature normalizada.
        """
        # Calcula estatísticas
        stats_exprs = []
        for col in cols:
            stats_exprs.extend([
                pl.col(col).median().alias(f"{col}_median"),
                pl.col(col).quantile(0.25).alias(f"{col}_q1"),
                pl.col(col).quantile(0.75).alias(f"{col}_q3"),
            ])
        stats = df.group_by(group_keys).agg(stats_exprs)
        # Faz join das estatísticas
        df = df.join(stats, on=group_keys, how="left")
        # Aplica normalização para radiant e dire
        for team_val, suffix in zip([0, 1], ["_radiant", "_dire"]):
            df = df.with_columns([
                pl.when(
                    pl.col("is_pick").eq(True) & pl.col("team").eq(team_val))
                .then(
                    (pl.col(col) - pl.col(f"{col}_median")) /
                    (pl.col(f"{col}_q3") - pl.col(f"{col}_q1"))
                ).otherwise(0)
                .alias(f"{col}{suffix}")
                for col in cols
            ])
        return df

    def filter(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        min_duration, max_duration = self.config.duration
        tiers = self.config.tier
        return lf.filter(
            pl.col("match_id").is_not_null(),
            pl.col("duration").is_between(min_duration, max_duration),
            pl.col("radiant_win").is_not_null(),
            pl.col("leagueid").is_not_null(),
            pl.col("game_mode").eq(CAPTAINS_MODE),
            pl.col("tier").is_in(tiers),
        )
