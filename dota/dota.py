import os
from dota.ai.autoencoder import Dota2AE
from dota.ai.cluster import Dota2Cluster
from dota.logger import get_logger
from dota.dataset.dataset import Dataset
import numpy as np
import pandas as pd

os.makedirs("log", exist_ok=True)
os.makedirs("tmp", exist_ok=True)
os.makedirs("loss_history", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("best", exist_ok=True)

log = get_logger("Dota2")


class Dota2():
    config = {
        "match_id": True,
        "start_time": True,
        "player_slot": True,
        "obs_placed": True,
        "sen_placed": True,
        "creeps_stacked": True,
        "camps_stacked": True,
        "rune_pickups": True,
        "firstblood_claimed": True,
        "teamfight_participation": True,
        "towers_killed": True,
        "roshans_killed": True,
        "stuns": True,
        "times": True,
        "gold_t": True,
        "lh_t": True,
        "dn_t": True,
        "xp_t": True,
        "party_id": True,
        "account_id": True,
        "hero_id": True,
        "items_vector": True,
        "backpack_vector": True,
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
        "version": True,
        "leagueid": True,
        "start_date_time": True,
        "duration": True,
        "patch": True,
        "region": True,
        "series_id": True,
        "series_type": True,
        "radiant_win": True,
        "tower_status_radiant": True,
        "tower_status_dire": True,
        "barracks_status_radiant": True,
        "barracks_status_dire": True,
        "leaguename": True,
        "tier": True,
        "gold_adv": True,
        "exp_adv": True,
        "hero_name": True,
        "attack_type": True,
        "roles_vector": True,
        "attack_range": True,
        "move_speed": True,
        "day_vision": True,
        "night_vision": True
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

    def __init__(self, year: int):
        ds = Dataset(year)
        self._dataset = ds

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
        log.info(f"Path: {self._dataset.cache_path}")
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
            sum([1 if self.config[key] else 0
                 for key in self.config.keys()]))

        dim += self.emb_pick_dim * 2 * self.n_players
        dim += self.emb_pick_dim * 2 * self.n_bans

        if (self.emb_config["roles_pick"]):
            dim += self.emb_role_dim * 2 * self.n_players
        if (self.emb_config["roles_ban"]):
            dim += self.emb_role_dim * 2 * self.n_bans
        if (self.emb_config["attributes"]):
            dim += self.emb_attr_dim * 2 * self.n_players
        if (self.emb_config["attack"]):
            dim += self.emb_attr_dim * 2 * self.n_players
        if (self.emb_config["items"]):
            dim += self.emb_item_dim * 2 * self.n_players
        if (self.emb_config["backpack"]):
            dim += self.emb_item_dim * 2 * self.n_players
        if (self.emb_config["neutral_items"]):
            dim += self.emb_neutral_dim * 2 * self.n_players
        return dim
