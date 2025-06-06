import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import polars as pl


class Dota2Autoencoder(nn.Module):
    def __init__(
        self,
        player_stats: list[str],
        hero_roles: list[str],
        dict_roles: dict[str, int],
        hero_pick_embedding_dim: int,
        hero_role_embedding_dim: int,
        n_heroes: int,
        n_players: int = 10,
        n_bans: int = 14,
        latent_dim: int = 32,
        hidden_layers: list[int] = [128, 64],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
    ):
        super(Dota2Autoencoder, self).__init__()
        # Configura o device para GPU se disponível, caso contrário CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Store the provided data
        self.game_stats = player_stats
        self.hero_stats = hero_roles
        self.dict_roles = dict_roles

        # Número de heróis, estatísticas, jogadores e bans
        self.n_heroes = n_heroes
        self.n_stats = len(hero_roles)  # Calculate from players_stats
        self.n_players = n_players
        self.n_bans = n_bans
        self.n_picks = n_players + n_bans
        # Calculate from dict_attributes
        self.n_roles = len(dict_roles)  # Calculate from dict_roles

        # Camadas de embedding para heróis, bans e estatísticas
        self.hero_pick_embedding = nn.Embedding(
            n_heroes + 1, hero_pick_embedding_dim, device=self.device)
        self.hero_role_embedding = nn.Embedding(
            self.n_roles, hero_pick_embedding_dim, device=self.device)

        # Dimensões e hiperparâmetros do modelo
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.hero_embedding_dim = hero_pick_embedding_dim
        self.hero_role_embedding_dim = hero_role_embedding_dim

        # Calcula a dimensão de entrada do modelo, para cada time
        self.input_dim = self.compute_input_dim()

        self.encoder = self.create_encoder()
        self.decoder = self.create_encoder(decoder=True)
        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        self.loss = nn.MSELoss()
        self.loss_history = []

    def compute_input_dim(self):
        picks_dim = self.n_players * self.hero_embedding_dim
        # Para cada ban: hero_embedding
        bans_dim = self.n_bans * self.hero_embedding_dim
        # Para cada jogador: stats
        stats_dim = self.n_players * self.n_stats
        hero_role_dim = self.n_players * self.hero_role_embedding_dim
        return picks_dim + bans_dim + stats_dim

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
                radiant_hero_roles: list[list[int]], dire_hero_roles: list[list[int]],
                radiant_stats: list[list[float]], dire_stats: list[list[float]],
                ) -> torch.Tensor:

        # Picks e bans
        radiant_picks_feat: torch.Tensor = self.hero_pick_embedding(
            torch.tensor(radiant_picks, device=self.device))
        radiant_bans_feat: torch.Tensor = self.hero_pick_embedding(
            torch.tensor(radiant_bans, device=self.device))
        dire_picks_feat: torch.Tensor = self.hero_pick_embedding(
            torch.tensor(dire_picks, device=self.device))
        dire_bans_feat: torch.Tensor = self.hero_pick_embedding(
            torch.tensor(dire_bans, device=self.device))

        radiant_hero_roles_feat = [
            self.hero_role_embedding(torch.tensor(
                roles, dtype=torch.long, device=self.device))
            for roles in radiant_hero_roles]
        dire_hero_roles_feat = [
            self.hero_role_embedding(torch.tensor(
                roles, dtype=torch.long, device=self.device))
            for roles in dire_hero_roles]

        radiant_stats_feat = [torch.tensor(stats, device=self.device) for stats in radiant_stats]
        dire_stats_feat = [torch.tensor(stats, device=self.device) for stats in dire_stats]

        # Flatten nested lists of tensors for stats and hero roles
        radiant_stats_feat = torch.cat(radiant_stats_feat, dim=0).flatten()
        dire_stats_feat = torch.cat(dire_stats_feat, dim=0).flatten()
        radiant_hero_roles_feat = torch.cat(radiant_hero_roles_feat, dim=0).flatten()
        dire_hero_roles_feat = torch.cat(dire_hero_roles_feat, dim=0).flatten()

        return torch.cat([
            radiant_picks_feat.flatten(), dire_picks_feat.flatten(),
            radiant_bans_feat.flatten(), dire_bans_feat.flatten(),
            radiant_stats_feat, dire_stats_feat,
            radiant_hero_roles_feat, dire_hero_roles_feat
        ], dim=0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encoda e decodifica os dados
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

    def encode(self, *args, **kwargs) -> torch.Tensor:
        latent, _ = self.forward(*args, **kwargs)
        return latent

    def train_step(self, radiant_picks: list[int], dire_picks: list[int],
                   radiant_bans: list[int], dire_bans: list[int],
                   radiant_hero_roles: list[list[int]], dire_hero_roles: list[list[int]],
                   radiant_stats: list[list[float]], dire_stats: list[list[float]]
                   ) -> float:
        self.train()
        self.optimizer.zero_grad()
        original = self.flatten(radiant_picks, dire_picks,
                                radiant_bans, dire_bans,
                                radiant_hero_roles, dire_hero_roles,
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

                radiant_hero_roles = row['radiant_hero_roles']
                dire_hero_roles = row['dire_hero_roles']

                radiant_stats: list[list[float]] = row['radiant_stats']
                dire_stats: list[list[float]]  = row['dire_stats']

                min_stats = np.array(row['min_stats'])
                max_stats = np.array(row['max_stats'])
                
                diff_stats = max_stats - min_stats
                diff_stats = np.where(diff_stats == 0, 1, diff_stats)  # Evita divisão por zero
                
                radiant_stats = [((np.array(stats) - min_stats) / diff_stats).tolist() for stats in radiant_stats]
                dire_stats = [((np.array(stats) - min_stats) / diff_stats).tolist() for stats in dire_stats]

                loss = self.train_step(
                    radiant_picks, dire_picks, radiant_bans, dire_bans,
                    radiant_hero_roles, dire_hero_roles,
                    radiant_stats, dire_stats)
                total_loss += loss
            avg_loss = total_loss / len(training_df)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
