import os
from dota.autoencoder.autoencoder import Dota2AE
from dota.autoencoder.cluster import Dota2Cluster
from dota.autoencoder.autoencoder import Dota2AE
from dota.autoencoder.cluster import Dota2Cluster
from dota.logger import get_logger, LogLevel
from dota.dataset.dataset import Dataset
import polars as pl
import numpy as np
import pandas as pd
import polars as pl
import numpy as np
import pandas as pd

os.makedirs("log", exist_ok=True)
os.makedirs("tmp", exist_ok=True)
os.makedirs("loss_history", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("best", exist_ok=True)


log = get_logger("Dota2", log_file="dota2.log")


class Dota2():
    config = {
        "creeps_stacked": False,
        "camps_stacked": False,
        "rune_pickups": False,
        "firstblood_claimed": False,
        "towers_killed": False,
        "roshans_killed": False,
        "stuns": False,
        "kills": True,
        "deaths": True,
        "assists": True,
        "last_hits": True,
        "denies": True,
        "gold_per_min": True,
        "xp_per_min": True,
        "level": True,
        "net_worth": True,
        "hero_damage": True,
        "tower_damage": True,
        "hero_healing": True,
        "total_gold": True,
        "total_xp": True,
        "neutral_kills": False,
        "tower_kills": False,
        "courier_kills": False,
        "hero_kills": False,
        "observer_kills": False,
        "sentry_kills": False,
        "roshan_kills": False,
        "necronomicon_kills": False,
        "ancient_kills": False,
        "buyback_count": False,
        "purchase_gem": False,
        "purchase_rapier": False,
    }

    embeddings_config = {
        "radiant_picks": True,
        "dire_picks": True,
        "radiant_bans": True,
        "dire_bans": True,
        "items": False,
        "backpack": False,
        "item_neutral": False,
        "roles_vector": False,
        "primary_attribute": False,
    }

    def __init__(self, year: int):
        ds = Dataset()
        self._dataset = ds
        self.path = f"{ds.data_path}/{year}"

        self.dict_attributes = len(self._dataset.dict_attributes) + 1
        self.dict_pick = len(self._dataset.dict_hero_index) + 1
        self.dict_ban = len(self._dataset.dict_hero_index) + 1
        self.dict_roles = len(self._dataset.dict_roles) + 1
        self.dict_attributes = len(self._dataset.dict_attributes) + 1
        self.dict_items = len(self._dataset.items_id) + 1

        self.emb_pick = 16
        self.emb_ban = 8
        self.emb_items = 16
        self.emb_role = 8
        self.emb_attributes = 8

        self.embeddings = [
            (self.dict_pick, self.emb_pick),
            (self.dict_pick, self.emb_pick),
            (self.dict_ban, self.emb_ban),
            (self.dict_ban, self.emb_ban),
            (self.dict_items, self.emb_items),
            (self.dict_items, self.emb_items),
            (self.dict_items, self.emb_items),
            (self.dict_roles, self.emb_role),
            (self.dict_attributes, self.emb_attributes),
        ]

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

        self.input_dim = self.calc_input_dim()

        self.latent_dim = 16
        self.encoder_layers = [self.input_dim, 512, 256, 128, 64, 32]
        self.decoder_layers = [32, 64, 128, 256, 512, self.input_dim]

        log.separator()
        log.info("Dota2 Initialized",)
        log.info(f"Configuration: {[k for k in self.config.keys() if self.config[k]]}")
        log.info(f"Embeddings: {[k for k in self.embeddings_config.keys() if self.embeddings_config[k]]}")
        log.info(f"Input Dimension: {self.input_dim}")
        log.info(f"Latent Dimension: {self.latent_dim}")
        log.info(f"Encoder Layers: {self.encoder_layers}")
        log.info(f"Decoder Layers: {self.decoder_layers}")
        log.info(f"Batch Size: {self.batch_size}")
        log.info(f"Learning Rate: {self.lr}")
        log.info(f"Epochs: {self.epochs}")
        log.info(f"Patience: {self.patience}")
        log.info(f"Path: {self.path}")
        log.info(f"Radiant Picks Embedding: {self.emb_pick}")
        log.info(f"Dire Picks Embedding: {self.emb_pick}")
        log.info(f"Radiant Bans Embedding: {self.emb_ban}")
        log.info(f"Dire Bans Embedding: {self.emb_ban}")
        log.info(f"Items Embedding: {self.emb_items}")
        log.info(f"Backpack Embedding: {self.emb_items}")
        log.info(f"Neutral Items Embedding: {self.emb_items}")
        log.info(f"Roles Vector Embedding: {self.emb_role}")
        log.info(f"Primary Attribute Embedding: {self.emb_attributes}")
        log.separator()


        self.ae = Dota2AE(
            name="Dota2AE",
            input_dim=self.calc_input_dim(),
            latent_dim=self.latent_dim,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            dropout=self.dropout,
            batch_size=self.batch_size,
            lr=self.lr,
            epochs=self.epochs,
            patience=self.patience,
            early_stopping=True,
            embeddings=self.embeddings,
            embeddings_config=self.embeddings_config,
        )
        self.ae.train_data(f"{self.path}/train.json",
                           f"{self.path}/val.json", self.config, self.embeddings_config)
        # self.cluster = Dota2Cluster()

    def calc_input_dim(self):
        dim = 10 * sum([1 if self.config[key] else 0 for key in self.config.keys()])
        dim += self.emb_pick * self.n_players if self.embeddings_config['radiant_picks'] else 0
        dim += self.emb_pick * self.n_players if self.embeddings_config['dire_picks'] else 0
        dim += self.emb_ban * self.n_bans if self.embeddings_config['radiant_bans'] else 0
        dim += self.emb_ban * self.n_bans if self.embeddings_config['dire_bans'] else 0
        dim += self.emb_items * self.n_items if self.embeddings_config['items'] else 0
        dim += self.emb_items * self.n_backpack if self.embeddings_config['backpack'] else 0
        dim += self.emb_items * self.n_neutral if self.embeddings_config['item_neutral'] else 0
        dim += self.emb_role * self.dict_roles if self.embeddings_config['roles_vector'] else 0
        dim += self.emb_attributes * self.dict_attributes if self.embeddings_config['roles_vector'] else 0
        return dim
