import polars as pl
from typing import Optional
from dota.logger import LogLevel, get_logger
from dota.dataset import Dota2Dataset
from dota.dota_schema import Dota2Schema
log = get_logger("Dota2", log_file='log/dota2.log')


class Dota2():

    def __init__(self, year: int, log_level: LogLevel = LogLevel.INFO):
        log.set_level(log_level)
        self.dataset = Dota2Dataset(year)
        
        self.schema = Dota2Schema(self.dataset, n_players=5, n_bans=7, n_items=6, n_backpack=3, n_neutral=1)

        self.emb = self.schema.emb_config
        self.input = self.schema.input

        self.dropout = 0.2
        self.batch_size = 64
        self.lr = 0.001
        self.epochs = 100
        self.patience = 10

        self.latent_dim = 16
        self.encoder_layers = [self.input, 512, 256, 128, 64, 32]
        self.decoder_layers = [32, 64, 128, 256, 512, self.input]

    def iterate(self, df: pl.DataFrame, batch_size: int = -1):
        if batch_size == -1:
            batch_size = self.batch_size

        log.separator()
        log.info("Iterating over dataset in batches")
        log.info(f"Batch size: {batch_size}")
        log.separator()
        return df.iter_slices(n_rows=batch_size)

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
