import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import polars as pl


class Dota2Autoencoder(nn.Module):
    def __init__(
        self,
        n_heroes: int,
        hero_embedding_dim: int,
        n_stats: int,
        team_embedding_dim: int,
        n_players: int = 10,
        n_bans: int = 14,
        n_attributes=6,
        n_roles=5,
        latent_dim: int = 32,
        hidden_layers: list[int] = [128, 64],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
    ):
        super(Dota2Autoencoder, self).__init__()
        # Configura o device para GPU se disponível, caso contrário CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Número de heróis, estatísticas, jogadores e bans
        self.n_heroes = n_heroes
        self.n_stats = n_stats
        self.n_players = n_players
        self.n_bans = n_bans
        self.n_picks = n_players + n_bans
        self.n_attributes = n_attributes  # Atributos primários dos heróis
        self.n_roles = n_roles  # Número de roles dos heróis

        # Camadas de embedding para heróis, bans e estatísticas
        self.hero_embedding = nn.Embedding(
            n_heroes, hero_embedding_dim, device=self.device)
        self.hero_attribute_embedding = nn.Embedding(
            n_attributes, hero_embedding_dim, device=self.device)
        self.hero_role_embedding = nn.Embedding(
            n_roles, hero_embedding_dim, device=self.device)
        self.team_embedding = nn.Embedding(
            2, team_embedding_dim, device=self.device)

        # Dimensões e hiperparâmetros do modelo
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.dropout = dropout

        # Calcula a dimensão de entrada do modelo, para cada time
        picks_dim = 2 * hero_embedding_dim
        bans_dim = 2 * team_embedding_dim
        stats_dim = n_players * n_stats
        self.input_dim = picks_dim + bans_dim + stats_dim

        self.encoder = self.create_encoder()
        self.decoder = self.create_encoder(decoder=True)
        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        self.loss = nn.MSELoss()
        self.loss_history = []

    def create_encoder(self, decoder: bool = False):
        layers = []
        hidden_layers = self.hidden_layers if not decoder else reversed(
            self.hidden_layers)
        dimension = self.input_dim if not decoder else self.latent_dim
        for _hidden in hidden_layers:
            layers.extend([
                nn.Linear(dimension, _hidden, device=self.device),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.BatchNorm1d(_hidden, device=self.device)
            ])
            dimension = _hidden
        layers.append(
            nn.Linear(dimension, self.latent_dim if not decoder else self.input_dim, device=self.device))
        return nn.Sequential(*layers).to(self.device)

    def flatten(self, radiant_picks: list[int], dire_picks: list[int],
                radiant_bans: list[int], dire_bans: list[int],
                radiant_stats: List[List[float]],
                dire_stats: List[List[float]]
                ) -> torch.Tensor:
        # Converte as listas de picks e bans em tensores
        radiant_picks_tensor = self.hero_embedding(
            torch.tensor(radiant_picks, device=self.device))
        dire_picks_tensor = self.hero_embedding(
            torch.tensor(dire_picks, device=self.device))
        radiant_bans_tensor = self.hero_embedding(
            torch.tensor(radiant_bans, device=self.device))
        dire_bans_tensor = self.hero_embedding(
            torch.tensor(dire_bans, device=self.device))

        # Converte as listas de estatísticas em tensores
        radiant_stats_flat = torch.tensor(
            radiant_stats, device=self.device).view(-1)
        dire_stats_flat = torch.tensor(dire_stats, device=self.device).view(-1)

        # Concatena os tensores de entrada
        return torch.cat([
            radiant_picks_tensor,
            dire_picks_tensor,
            radiant_bans_tensor,
            dire_bans_tensor,
            radiant_stats_flat,
            dire_stats_flat
        ], dim=0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoda e decodifica os dados
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

    def encode(self, *args, **kwargs) -> torch.Tensor:
        latent, _ = self.forward(*args, **kwargs)
        return latent

    def train_step(self, radiant_picks: list[int], dire_picks: list[int],
                   radiant_bans: list[int], dire_bans: list[int],
                   radiant_stats: List[List[float]],
                   dire_stats: List[List[float]]
                   ) -> float:
        self.train()
        self.optimizer.zero_grad()
        original = self.flatten(radiant_picks, dire_picks,
                                radiant_bans, dire_bans,
                                radiant_stats, dire_stats)
        latent, reconstructed = self.forward(original)
        loss = self.loss(original, reconstructed)
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(loss.item())
        return loss.item()

    def train_data(self, training_df: pl.DataFrame, epochs: int = 10) -> None:
        for epoch in range(epochs):
            total_loss = 0.0
            for row in training_df.iter_rows(named=True):
                radiant_picks = row['radiant_picks']
                dire_picks = row['dire_picks']
                radiant_bans = row['radiant_bans']
                dire_bans = row['dire_bans']
                radiant_stats = row['radiant_stats']
                dire_stats = row['dire_stats']

                loss = self.train_step(
                    radiant_picks, dire_picks, radiant_bans, dire_bans, radiant_stats, dire_stats)
                total_loss += loss

            avg_loss = total_loss / len(training_df)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
