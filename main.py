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
    if("--save" in sys.argv):
        save_dataset(dataset.head())
        return
    
    heroes, _, dict_roles = get_heroes(path)
    n_heroes = heroes.select("hero_id").max().collect().item()
    print(f"Total Heroes: {n_heroes}, Total Roles: {len(dict_roles)}")
    autoencoder = Dota2Autoencoder(
        player_stats=player_cols,
        hero_roles=hero_cols,
        dict_roles=dict_roles,
        hero_pick_embedding_dim=16,
        hero_role_embedding_dim=8,
        n_heroes=n_heroes,
        n_players=5,
        n_bans=7,
        latent_dim=32,
        hidden_layers=[128, 64],
        dropout=0.2,
        learning_rate=0.001
    )
    
    print("Training Dota2 Autoencoder...")
    
    autoencoder.train_data(dataset, 10)
    
    print("Training completed.")
if __name__ == "__main__":
    main()
