import kagglehub
import polars as pl
import numpy as np
from dataset import get_dataset, save_dataset
from model import Dota2Autoencoder
from typing import Dict, List, Tuple


def main():
    # Download do dataset
    dataset_name = "bwandowando/dota-2-pro-league-matches-2023"
    path = kagglehub.dataset_download(dataset_name)


    dataset = get_dataset(path)

    # autoencoder = Dota2Autoencoder(
    #     n_heroes=n_heroes,
    #     hero_embedding_dim=16,
    #     n_stats=len(game_cols),
    #     team_embedding_dim=8,
    #     n_players=10,
    #     n_picks=14,
    #     latent_dim=32,
    #     hidden_layers=[128, 64],
    #     dropout=0.2,
    #     learning_rate=0.001
    # )

    print(dataset)


if __name__ == "__main__":
    main()
