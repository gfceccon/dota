import os
from dota.autoencoder.autoencoder import Dota2AE
from dota.autoencoder.cluster import Dota2Cluster
from dota.logger import get_logger, LogLevel
from dota.dataset.dataset import Dataset
import polars as pl
import numpy as np
import pandas as pd

os.makedirs("log", exist_ok=True)
os.makedirs("tmp", exist_ok=True)
os.makedirs("loss_history", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("best", exist_ok=True)


log = get_logger("Dota2", LogLevel.INFO)


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
    }

    def __init__(self, year: int):
        ds = Dataset()
        self._dataset = ds
        self.path = f"{ds.data_path}/{year}"

        self.dict_attr = len(self._dataset.dict_attributes) + 1
        self.dict_pick = len(self._dataset.dict_hero_index) + 1
        self.dict_ban = len(self._dataset.dict_hero_index) + 1
        self.dict_role = len(self._dataset.dict_roles) + 1
        self.dict_item = len(self._dataset.items_id) + 1

        self.emb_pick = 16
        self.emb_ban = 8
        self.emb_items = 16
        self.emb_role = 8

        self.embeddings = [
            (self.dict_pick, self.emb_pick),
            (self.dict_pick, self.emb_pick),
            (self.dict_ban, self.emb_ban),
            (self.dict_ban, self.emb_ban),
            (self.dict_item, self.emb_items),
            (self.dict_item, self.emb_items),
            (self.dict_item, self.emb_items),
            (self.dict_role, self.emb_role),
        ]

        self.input_dim = self.calc_input_dim()

        self.latent_dim = 16
        self.encoder_layers = [self.input_dim, 512, 256, 128, 64, 32]
        self.decoder_layers = [32, 64, 128, 256, 512, self.input_dim]

        self.dropout = 0.2
        self.batch_size = 64
        self.lr = 0.001
        self.epochs = 100
        self.patience = 10

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
        log.info(f"Modelo Dota2AE criado com input_dim={self.input_dim}, latent_dim={self.latent_dim}, encoder_layers={self.encoder_layers}, decoder_layers={self.decoder_layers}, dropout={self.dropout}, batch_size={self.batch_size}, lr={self.lr}, epochs={self.epochs}, patience={self.patience}")
        self.ae.train_data(f"{self.path}/train.json",
                           f"{self.path}/val.json", self.config, self.embeddings_config)
        # self.cluster = Dota2Cluster()

    def calc_input_dim(self):
        dim = 10 * sum([1 if self.config[key] else 0 for key in self.config])
        dim += self.emb_pick if self.embeddings_config['radiant_picks'] else 0
        dim += self.emb_pick if self.embeddings_config['dire_picks'] else 0
        dim += self.emb_ban if self.embeddings_config['radiant_bans'] else 0
        dim += self.emb_ban if self.embeddings_config['dire_bans'] else 0
        dim += self.emb_items if self.embeddings_config['items'] else 0
        dim += self.emb_items if self.embeddings_config['backpack'] else 0
        dim += self.emb_items if self.embeddings_config['item_neutral'] else 0
        dim += self.emb_role if self.embeddings_config['roles_vector'] else 0
        return dim

    def ae_create(self):
        ...

    def ae_train(self):
        ...

    def ae_predict_ti(self, year: int):
        ...
