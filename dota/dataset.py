import os
import kagglehub
import polars as pl
from .schemas import Schema
from dota.logger import LogLevel, get_logger
from dota.dataset_helper import DatasetHelper

# Inicializa o logger para registrar informações e erros do dataset
log = get_logger(name='Dota Dataset',  log_file='log/dota_dataset.log',)

# Nome do dataset no KaggleHub
DATASET = 'bwandowando/dota-2-pro-league-matches-2023'
# Código do modo de jogo 'Capitães'
CAPTAINS_MODE = 2


class Dataset:
    """
    Classe principal para manipulação e processamento do dataset de partidas profissionais de Dota 2.
    Responsável por carregar, filtrar, processar e salvar os dados em diferentes formatos.
    """
    # Tipos dos atributos principais (LazyFrames do Polars)
    heroes_lf: 'pl.LazyFrame'
    picks_bans_lf: 'pl.LazyFrame'
    leagues_lf: 'pl.LazyFrame'
    objectives_lf: 'pl.LazyFrame'
    players_lf: 'pl.LazyFrame'
    metadata_lf: 'pl.LazyFrame'
    metadata_df: 'pl.DataFrame'

    def __init__(self, year: int,
                 path: str = '',
                 log_level=LogLevel.INFO,
                 force_download: bool = False):
        """
        Inicializa o objeto Dataset, configurando caminhos, logger e otimizando queries.
        year: Ano dos dados a serem carregados.
        path: Caminho base dos dados (opcional).
        log_level: Nível de log desejado.
        force_download: Se True, força o download dos dados.
        """
        log.set_level(log_level)
        log.separator()
        log.info("Inicializando Dataset Dota 2...")

        self.config = DatasetHelper(path, force_download=force_download)
        self.cache_path = f"{self.config.data_path}/{year}"
        self.year = year
        self.check_year()
        self.config.load()
        # Otimizações para consultas com Polars
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

        log.info(f"Carregando dados do ano {self.year}...")

        # Cria diretório de cache se não existir
        if not os.path.exists(self.cache_path):
            log.info(
                f"Cache não encontrado para o ano {self.year}, criando diretório...")
            os.makedirs(self.cache_path, exist_ok=True)

        # Dicionário com os arquivos e funções de fallback para cada tipo de dado
        files = {
            'heroes_lf': ('heroes.parquet', lambda: self.config._heroes()),
            'picks_bans_lf': ('picks_bans.parquet', lambda: self.config._picks_bans(self.year)),
            'leagues_lf': ('leagues.parquet', lambda: self.config._leagues.lazy()),
            'objectives_lf': ('objectives.parquet', lambda: self.config._objectives(self.year)),
            'players_lf': ('players.parquet', lambda: self.config._players(self.year)),
            'metadata_lf': ('metadata.parquet', lambda: self.config._metadata(self.year)),
        }

        # Carrega os arquivos do cache ou gera e salva se não existirem
        for attr, (fname, fallback) in files.items():
            fpath = os.path.join(self.cache_path, fname)
            if os.path.exists(fpath):
                setattr(self, attr, pl.scan_parquet(fpath))
            else:
                data = fallback()
                data.collect(
                    optimizations=self.optimizations).write_parquet(fpath)
                setattr(self, attr, data)

    def get(self, path: str = '') -> str:
        """
        Retorna o caminho do arquivo principal de dados (data.parquet), gerando-o se necessário.
        path: Caminho alternativo para salvar/carregar o arquivo.
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
            log.separator()
            return data_parquet_path
        if (not path):
            log.info(
                f"Cache não encontrado para o ano {self.year}, processando dados...")

        os.makedirs(cache_dir, exist_ok=True)
        log.separator()

        log.info("Salvando arquivo principal...")
        games = self.games()
        data = games.collect(optimizations=self.optimizations)
        data.write_parquet(data_parquet_path)
        log.info(f"Cache salvo em {cache_dir}")
        return data_parquet_path

    def metadata(self) -> pl.LazyFrame:
        """
        Gera e retorna os metadados das partidas, unindo informações de ligas, ouro e experiência.
        """
        self.check_year()
        log.separator()
        log.info(f"Criando metadados...")
        log.separator()
        cols = self.config.metadata_cols()

        metadata = self.filter((
            self.metadata_lf.unique(subset=["match_id"])
            .join(self.leagues_lf, on="leagueid", how="inner")
        ))

        metadata = (
            metadata
            .join(self.gold(), on="match_id",)
            .join(self.exp(), on="match_id",)
            .select(cols)
        )
        return metadata

    def players(self) -> pl.LazyFrame:
        """
        Processa e retorna informações detalhadas dos jogadores, picks e bans, itens e heróis.
        """
        self.check_year()
        log.separator()
        log.info(f"Carregando dados dos jogadores...")
        log.separator()

        player_cols = self.config.players_cols()
        ban_cols = self.config.ban_cols()

        players_lf = self.filter(
            self.players_lf
            .join(self.metadata_lf, on="match_id", how="inner")
            .join(self.leagues_lf, on="leagueid", how="inner")
        )

        players_picks_bans = (
            players_lf
            .join(self.picks_bans_lf, on=["match_id", "hero_id"], how="right")
        )

        players_items = self.items_backpack(players_picks_bans)

        players_heroes = (
            players_items
            .join(self.heroes_lf, on="hero_id", how="inner")
        )

        players = (
            players_heroes
            .with_columns([
                *[
                    pl.when(pl.col("is_pick").eq(True)
                            & pl.col("team").eq(team))
                    .then(pl.col(col))
                    .alias(f"{name}")
                    for col, name, team in player_cols
                ],
                *[
                    pl.when(pl.col("is_pick").eq(False)
                            & pl.col("team").eq(team))
                    .then(pl.col(col))
                    .alias(f"{name}")
                    for col, name, team in ban_cols
                ],
            ])
        )

        players = (
            players
            .select(["match_id"] + [name for _, name, _ in player_cols] + [name for _, name, _ in ban_cols]))

        return players

    def games(self) -> pl.LazyFrame:
        """
        Retorna um LazyFrame com as informações agregadas das partidas, unindo metadados e jogadores.
        """
        cols = self.config.games_cols()
        cols.remove("match_id")

        data = (
            self.metadata()
            .join(self.players(), on="match_id", how="inner")
            .group_by("match_id")
            .agg(cols)
        )
        return data

    def gold(self) -> pl.LazyFrame:
        """
        Carrega e agrega os dados de vantagem de ouro por minuto para cada partida.
        """
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
        """
        Carrega e agrega os dados de vantagem de experiência por minuto para cada partida.
        """
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
        """
        Converte os itens e mochilas dos jogadores para índices e cria vetores para análise.
        data: LazyFrame de jogadores e picks/bans.
        """
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
        """
        Verifica se o ano selecionado é válido para o dataset.
        """
        self.config.check_year(self.year)

    @staticmethod
    def force_download() -> str:
        """
        Força o download do dataset completo do KaggleHub.
        :return: Caminho do dataset baixado.
        """
        log.separator()
        log.info("Forçando download do dataset completo...")
        return kagglehub.dataset_download(handle=DATASET, force_download=True,)

    @staticmethod
    def save_year(year: int, path: str = '') -> str:
        """
        Salva o dataset do ano especificado em um arquivo Parquet.
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

    def filter(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Aplica filtros padrão para garantir a qualidade dos dados das partidas.
        """
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
