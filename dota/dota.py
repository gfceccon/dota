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
        "item_neutral": True,
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
        "neutral_kills": True,
        "tower_kills": True,
        "courier_kills": True,
        "hero_kills": True,
        "observer_kills": True,
        "sentry_kills": True,
        "roshan_kills": True,
        "necronomicon_kills": True,
        "ancient_kills": True,
        "buyback_count": True,
        "purchase_gem": True,
        "purchase_rapier": True,
    }

    emb_config = {
        "radiant_picks": True,
        "dire_picks": True,
        "radiant_bans": True,
        "dire_bans": True,

        "roles_pick": False,
        "roles_ban": False,
        "attributes": False,
        "attack": False,
        "items": False,
        "backpack": False,
        "neutral_items": False,
    }

    columns = {
        "player_radiant_stats": True,
        "player_dire_stats": True,
    }

    def __init__(self, year: int):
        ds = Dataset()
        self._dataset = ds
        self.path = f"{ds.data_path}/{year}"

        self.dict_attributes = len(ds.dict_attributes) + 1
        self.dict_pick = len(ds.dict_hero_index) + 1
        self.dict_ban = len(ds.dict_hero_index) + 1
        self.dict_roles = len(ds.dict_roles) + 1
        self.dict_items = len(ds.items_id) + 1

        self.emb_pick = 32
        self.emb_ban = 32
        self.emb_items = 16
        self.emb_role = 16
        self.emb_attributes = 8
        self.emb_neutral = 8

        self.embeddings = {
            "radiant_picks": (self.emb_pick, self.dict_pick),
            "dire_picks": (self.emb_pick, self.dict_pick),
            "radiant_bans": (self.emb_ban, self.dict_ban),
            "dire_bans": (self.emb_ban, self.dict_ban),


            "radiant_roles_picks": (self.emb_role, self.dict_roles),
            "dire_roles_picks": (self.emb_role, self.dict_roles),
            "radiant_roles_bans": (self.emb_role, self.dict_roles),
            "dire_roles_bans": (self.emb_role, self.dict_roles),

            "radiant_attributes": (self.emb_attributes, self.dict_attributes),
            "dire_attributes": (self.emb_attributes, self.dict_attributes),
            "radiant_attack": (self.emb_attributes, self.dict_attributes),
            "dire_attack": (self.emb_attributes, self.dict_attributes),

            "radiant_items": (self.emb_items, self.dict_items),
            "dire_items": (self.emb_items, self.dict_items),

            "radiant_backpack": (self.emb_items, self.dict_items),
            "dire_backpack": (self.emb_items, self.dict_items),

            "radiant_neutral_items": (self.emb_neutral, self.dict_items),
            "dire_neutral_items": (self.emb_neutral, self.dict_items),
        }

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
        log.info(
            f"Configuration: {[k for k in self.config.keys() if self.config[k]]}")
        log.info(
            f"Embeddings: {[k for k in self.emb_config.keys() if self.emb_config[k]]}")
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
            embeddings_config=self.embeddings,
        )

        self.ae.train_data(
            f"{self.path}/train.json",
            f"{self.path}/val.json",
            self.columns,
            "player_radiant_stats",
            "player_dire_stats",
            self.emb_config)
        # self.cluster = Dota2Cluster()

    def calc_input_dim(self):
        dim = (
            2 * self.n_players *
            sum([1 if self.config[key] else 0
                 for key in self.config.keys()]))

        dim += self.emb_pick * 2 * self.n_players
        dim += self.emb_ban * 2 * self.n_bans

        if(self.emb_config["roles_pick"]):
            dim += self.emb_role * 2 * self.n_players
        if(self.emb_config["roles_ban"]):
            dim += self.emb_role * 2 * self.n_bans
        if(self.emb_config["attributes"]):
            dim += self.emb_attributes * 2 * self.n_players
        if(self.emb_config["attack"]):
            dim += self.emb_attributes * 2 * self.n_players
        if(self.emb_config["items"]):
            dim += self.emb_items * 2 * self.n_players
        if(self.emb_config["backpack"]):
            dim += self.emb_items * 2 * self.n_players
        if(self.emb_config["neutral_items"]):
            dim += self.emb_neutral * 2 * self.n_players
        return dim
