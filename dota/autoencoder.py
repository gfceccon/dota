import csv
from typing import Any, cast
import torch.nn as nn
from torch import Tensor
import numpy as np
import pandas as pd
import torch
from dota.logger import get_logger

log = get_logger("Dota2AE")


class Dota2AE(nn.Module):
    def __init__(
        self,
        name: str,
        lr: float,
        dropout: float,
        early_stopping: bool,
        epochs: int,
        patience: int,
        batch_size: int,
        input_dim: int,
        latent_dim: int,
        encoder_layers: list[int],
        decoder_layers: list[int],
        embeddings_config: dict[str, tuple[int, int]],
    ):
        super(Dota2AE, self).__init__()

        # Nome do modelo
        self.name = f"Dota2AE_{name}"
        self.best_filename = f"Dota2AE_{name}_best.h5"
        self.loss_path = f"./Dota2AE_{name}_loss.csv"

        # Camadas de embedding para heróis, bans e estatísticas
        self.embeddings = {
            name: nn.Embedding(emb[1], emb[0], device=self.device)
            for name, emb in embeddings_config.items()
        }
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.embeddings_config = embeddings_config

        # Dimensões e hiperparâmetros do modelo
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.learning_rate = lr
        self.dropout = dropout
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.early_stopping = early_stopping

        # Históricos de loss
        self.train_stopped = 0
        self.loss_val_history = []
        self.loss_history = []
        self.best_val_loss = float('inf')
