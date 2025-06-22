import os
from typing import Optional
import polars as pl
from dota.autoencoder import Dota2AE
from dota.cluster import Dota2Cluster
from dota.logger import LogLevel, get_logger
from dota.dataset import Dataset
from dota.dota_columns import columns, columns_emb, embeddings, calc_input_dim
log = get_logger("Dota2", log_file='log/dota2.log')


class Dota2():

    def __init__(self, year: int, log_level: LogLevel = LogLevel.INFO):
        log.set_level(log_level)
        ds = Dataset(year)
        self.dataset = ds

        # Convert dict[int, int] to dict[str, int]
        self.emb_attr = [v for k, v in ds.config.attr_mapping.items()]
        self.emb_role = [v for k, v in ds.config.role_mapping.items()]
        self.emb_item = [v for k, v in ds.config.item_mapping.items()]
        self.emb_hero = [v for k, v in ds.config.hero_mapping.items()]

        self.emb_pick_dim = 32
        self.emb_ban_dim = 32
        self.emb_item_dim = 16
        self.emb_role_dim = 16
        self.emb_attr_dim = 8
        self.emb_neutral_dim = 8

        emb_config: dict[str, list[int]] = {
            'emb_attr': self.emb_attr,
            'emb_role': self.emb_role,
            'emb_item': self.emb_item,
            'emb_hero': self.emb_hero,
        }
        emb_dim_config: dict[str, int] = {
            'emb_pick_dim': self.emb_pick_dim,
            'emb_ban_dim': self.emb_ban_dim,
            'emb_item_dim': self.emb_item_dim,
            'emb_role_dim': self.emb_role_dim,
            'emb_attr_dim': self.emb_attr_dim,
            'emb_neutral_dim': self.emb_neutral_dim,
        }

        self.emb = embeddings(emb_config, emb_dim_config)
        self.input_dim = calc_input_dim(5, 7, emb_config)

        self.dropout = 0.2
        self.batch_size = 64
        self.lr = 0.001
        self.epochs = 100
        self.patience = 10

        self.n_players = 5
        self.n_bans = 7
        self.n_total_npc = 2 * (self.n_players + self.n_bans)
        self.n_items = 6
        self.n_backpack = 3
        self.n_neutral = 1

        self.latent_dim = 16
        self.encoder_layers = [self.input_dim, 512, 256, 128, 64, 32]
        self.decoder_layers = [32, 64, 128, 256, 512, self.input_dim]

    def log_config(self):
        log.separator()
        log.info("Dota2 Initialized",)
        log.info(
            f"Configuration: {[k for k in columns.keys() if columns[k]]}")
        log.info(
            f"Embeddings: {[k for k in columns_emb.keys() if columns_emb[k]]}")
        log.info(f"Input Dimension: {self.input_dim}")
        log.info(f"Latent Dimension: {self.latent_dim}")
        log.info(f"Encoder Layers: {self.encoder_layers}")
        log.info(f"Decoder Layers: {self.decoder_layers}")
        log.info(f"Batch Size: {self.batch_size}")
        log.info(f"Learning Rate: {self.lr}")
        log.info(f"Epochs: {self.epochs}")
        log.info(f"Patience: {self.patience}")
        log.info(f"Cache Path: {self.dataset.cache_path}")
        log.info(f"Radiant Picks Embedding: {self.emb_pick_dim}")
        log.info(f"Dire Picks Embedding: {self.emb_pick_dim}")
        log.info(f"Radiant Bans Embedding: {self.emb_ban_dim}")
        log.info(f"Dire Bans Embedding: {self.emb_ban_dim}")
        log.info(f"Items Embedding: {self.emb_item_dim}")
        log.info(f"Backpack Embedding: {self.emb_item_dim}")
        log.info(f"Neutral Items Embedding: {self.emb_item_dim}")
        log.info(f"Roles Vector Embedding: {self.emb_role}")
        log.info(f"Primary Attribute Embedding: {self.emb_attr_dim}")
        log.separator()

    def iterate(self, df: Optional[pl.DataFrame] = None, batch_size: int = -1):
        if batch_size == -1:
            batch_size = self.batch_size
        if df is None:
            log.info("No DataFrame provided, loading from dataset")
            dataset = self.dataset.get()
            lf = pl.scan_parquet(dataset)
            df = lf.collect(optimizations=self.dataset.optimizations)

        log.separator()
        log.info("Iterating over dataset in batches")
        log.info(f"Batch size: {batch_size}")
        log.separator()
        return df.iter_slices(n_rows=batch_size)

    def all(self) -> pl.DataFrame:
        log.separator()
        log.info("Loading entire dataset")
        dataset = self.dataset.get()
        lf = pl.scan_parquet(dataset)
        df = lf.collect(optimizations=self.dataset.optimizations)
        return df

    def split(self, train_size=0.7, validation_size=0.15, test_size=0.15) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        if (train_size + validation_size + test_size) != 1.0:
            log.error("Train, validation, and test sizes do not sum to 1.0.")
            raise ValueError(
                "Train, validation, and test sizes must sum to 1.0")
        log.separator()
        log.info("Splitting dataset into train, validation, and test sets")
        dataset = self.dataset.get()
        lf = pl.scan_parquet(dataset)
        df = lf.collect(optimizations=self.dataset.optimizations)
        train = df.sample(fraction=0.7, seed=42, shuffle=True)
        valid = df.sample(fraction=0.15, seed=42, shuffle=True)
        test = df.sample(fraction=0.15, seed=42, shuffle=True)
        return train, valid, test
