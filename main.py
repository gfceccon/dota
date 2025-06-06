import kagglehub
import polars as pl
import numpy as np
import json
import torch
from dataset import get_dataset, save_dataset
from heroes import get_heroes
from model import Dota2Autoencoder
from typing import Dict, List, Tuple
import sys


def main():
    # Carregar metadados dos heróis
    dataset_name = "bwandowando/dota-2-pro-league-matches-2023"
    path = kagglehub.dataset_download(dataset_name)
    dataset, player_cols, hero_cols = get_dataset(path)
    
    heroes, _, dict_roles = get_heroes(path)
    n_heroes = heroes.select("hero_id").max().collect().item()
    for roles in dict_roles:
        dict_roles[roles] = dict_roles[roles]
    n_roles = max(dict_roles.values())
    n_hero_stats = len(hero_cols)
    n_player_stats = len(player_cols)
    print (f"Total Hero Stats: {n_hero_stats}, Total Player Stats: {n_player_stats}")
    print(f"Total Roles: {n_roles}, Total Heroes: {n_heroes}")
    print(f"Total Heroes: {n_heroes}, Total Roles: {len(dict_roles)}")

    if("--save" in sys.argv):
        save_dataset(dataset.head())
        return
    
    autoencoder = Dota2Autoencoder(
        hero_pick_embedding_dim=16,
        hero_role_embedding_dim=8,
        n_player_stats=n_player_stats,
        n_heroes=n_heroes + 1,
        n_roles=n_roles + 1,
        n_players=5,
        n_bans=7,
        latent_dim=32,
        hidden_layers=[128, 64],
        dropout=0.2,
        learning_rate=0.001,
        verbose=True
    )
    
    print("Training Dota2 Autoencoder...")
    
    autoencoder.train_data(dataset, 10, verbose=False)
    
    print("Training completed.")
if __name__ == "__main__":
    main()
