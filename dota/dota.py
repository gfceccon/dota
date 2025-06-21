import os
from typing import Optional
import polars as pl
from dota.autoencoder import Dota2AE
from dota.cluster import Dota2Cluster
from dota.logger import LogLevel, get_logger
from dota.dataset import Dataset

log = get_logger("Dota2", log_file='log/dota2.log')


class Dota2():
    columns = {
        "match_id": False,
        'player_slot': False,
        "start_time": False,
        "player_slot": False,
        "obs_placed": True,
        "sen_placed": True,
        "creeps_stacked": True,
        "camps_stacked": True,
        "rune_pickups": True,
        "firstblood_claimed": False,
        "teamfight_participation": True,
        "towers_killed": True,
        "roshans_killed": True,
        "stuns": True,
        "times": False,
        "gold_t": False,
        "lh_t": False,
        "dn_t": False,
        "xp_t": False,
        "party_id": False,
        "account_id": False,
        "hero_id": True,
        "items_vector": False,
        "backpack_vector": False,
        "item_neutral": False,
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
        "gold": True,
        "gold_spent": True,
        "neutral_kills": True,
        "tower_kills": True,
        "courier_kills": True,
        "lane_kills": True,
        "hero_kills": True,
        "observer_kills": True,
        "sentry_kills": True,
        "roshan_kills": True,
        "necronomicon_kills": True,
        "ancient_kills": True,
        "buyback_count": True,
        "purchase_ward_observer": True,
        "purchase_ward_sentry": True,
        "purchase_gem": True,
        "purchase_rapier": True,
        "is_pick": True,
        "team": True,
        "order": True,
        "version": False,
        "leagueid": False,
        "start_date_time": False,
        "duration": False,
        "patch": False,
        "region": False,
        "series_id": False,
        "series_type": False,
        "radiant_win": False,
        "tower_status_radiant": False,
        "tower_status_dire": False,
        "barracks_status_radiant": False,
        "barracks_status_dire": False,
        "leaguename": False,
        "tier": False,
        "gold_adv": False,
        "exp_adv": False,
        "hero_name": False,
        "attack_type": False,
        "roles_vector": False,
        "attack_range": False,
        "move_speed": False,
        "day_vision": False,
        "night_vision": False
    }

    columns_emb = {
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

    def __init__(self, year: int, log_level: LogLevel = LogLevel.INFO):
        log.set_level(log_level)
        ds = Dataset(year)
        self.dataset = ds

        self.emb_attr = ds.config.attr_mapping
        self.emb_role = ds.config.role_mapping
        self.emb_item = ds.config.item_mapping
        self.emb_hero = ds.config.hero_mapping

        self.emb_pick_dim = 32
        self.emb_ban_dim = 32
        self.emb_item_dim = 16
        self.emb_role_dim = 16
        self.emb_attr_dim = 8
        self.emb_neutral_dim = 8

        self.embeddings = {
            "radiant_picks": (self.emb_hero, self.emb_pick_dim),
            "dire_picks": (self.emb_hero, self.emb_pick_dim),
            "radiant_bans": (self.emb_hero, self.emb_ban_dim),
            "dire_bans": (self.emb_hero, self.emb_ban_dim),


            "radiant_roles_picks": (self.emb_role, self.emb_role_dim),
            "dire_roles_picks": (self.emb_role, self.emb_role_dim),
            "radiant_roles_bans": (self.emb_role, self.emb_role_dim),
            "dire_roles_bans": (self.emb_role, self.emb_role_dim),

            "radiant_attributes": (self.emb_attr, self.emb_attr_dim),
            "dire_attributes": (self.emb_attr, self.emb_attr_dim),
            "radiant_attack": (self.emb_attr, self.emb_attr_dim),
            "dire_attack": (self.emb_attr, self.emb_attr_dim),

            "radiant_items": (self.emb_item, self.emb_item_dim),
            "dire_items": (self.emb_item, self.emb_item_dim),

            "radiant_backpack": (self.emb_item, self.emb_item_dim),
            "dire_backpack": (self.emb_item, self.emb_item_dim),

            "radiant_neutral_items": (self.emb_item, self.emb_neutral_dim),
            "dire_neutral_items": (self.emb_item, self.emb_neutral_dim),
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

    def log_config(self):
        log.separator()
        log.info("Dota2 Initialized",)
        log.info(
            f"Configuration: {[k for k in self.columns.keys() if self.columns[k]]}")
        log.info(
            f"Embeddings: {[k for k in self.columns_emb.keys() if self.columns_emb[k]]}")
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

    def calc_input_dim(self):
        dim = (
            2 * self.n_players *
            sum([1 if self.columns[key] else 0
                 for key in self.columns.keys()]))

        dim += self.emb_pick_dim * 2 * self.n_players
        dim += self.emb_pick_dim * 2 * self.n_bans

        if (self.columns_emb["roles_pick"]):
            dim += self.emb_role_dim * 2 * self.n_players
        if (self.columns_emb["roles_ban"]):
            dim += self.emb_role_dim * 2 * self.n_bans
        if (self.columns_emb["attributes"]):
            dim += self.emb_attr_dim * 2 * self.n_players
        if (self.columns_emb["attack"]):
            dim += self.emb_attr_dim * 2 * self.n_players
        if (self.columns_emb["items"]):
            dim += self.emb_item_dim * 2 * self.n_players
        if (self.columns_emb["backpack"]):
            dim += self.emb_item_dim * 2 * self.n_players
        if (self.columns_emb["neutral_items"]):
            dim += self.emb_neutral_dim * 2 * self.n_players
        return dim

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
        if(train_size + validation_size + test_size) != 1.0:
            log.error("Train, validation, and test sizes do not sum to 1.0.")
            raise ValueError("Train, validation, and test sizes must sum to 1.0")
        log.separator()
        log.info("Splitting dataset into train, validation, and test sets")
        dataset = self.dataset.get()
        lf = pl.scan_parquet(dataset)
        df = lf.collect(optimizations=self.dataset.optimizations)
        train = df.sample(fraction=0.7, seed=42, shuffle=True)
        valid = df.sample(fraction=0.15, seed=42, shuffle=True)
        test = df.sample(fraction=0.15, seed=42, shuffle=True)
        return train, valid, test
