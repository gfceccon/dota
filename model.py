import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import polars as pl


class Dota2Autoencoder(nn.Module):
    def __init__(
        self,
        hero_pick_embedding_dim: int,
        hero_role_embedding_dim: int,
        n_player_stats: int,
        n_roles: int,
        n_heroes: int,
        n_players: int = 10,
        n_bans: int = 14,
        latent_dim: int = 32,
        hidden_layers: list[int] = [128, 64],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        verbose: bool = False
    ):
        super(Dota2Autoencoder, self).__init__()
        # Configura o device para GPU se disponível, caso contrário CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Número de heróis, estatísticas, jogadores e bans
        self.n_heroes = n_heroes
        self.n_player_stats = n_player_stats
        self.n_players = n_players
        self.n_bans = n_bans
        self.n_picks = n_players + n_bans
        self.n_roles = n_roles

        # Camadas de embedding para heróis, bans e estatísticas
        self.hero_pick_embedding = nn.Embedding(
            n_heroes, hero_pick_embedding_dim, device=self.device)
        self.hero_role_embedding = nn.Embedding(
            self.n_roles, hero_role_embedding_dim, device=self.device)

        # Dimensões e hiperparâmetros do modelo
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.hero_embedding_dim = hero_pick_embedding_dim
        self.hero_role_embedding_dim = hero_role_embedding_dim

        # Calcula a dimensão de entrada do modelo, para cada time
        self.input_dim = self.compute_input_dim()

        self.encoder = self.create_encoder(verbose=verbose)
        self.decoder = self.create_encoder(decoder=True, verbose=verbose)
        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        self.loss = nn.MSELoss()
        self.loss_history = []

    def compute_input_dim(self):
        picks_dim = 2 * self.n_players * self.hero_embedding_dim
        bans_dim = 2 * self.n_bans * self.hero_embedding_dim
        stats_dim = 2 * self.n_players * self.n_player_stats
        hero_role_dim = 2 * self.n_players * self.n_roles * self.hero_role_embedding_dim
        self.input_dim = picks_dim + bans_dim + stats_dim + hero_role_dim
        return self.input_dim

    def create_encoder(self, decoder: bool = False, verbose: bool = False) -> nn.Sequential:
        layers = []
        hidden_layers = self.hidden_layers if not decoder else reversed(
            self.hidden_layers)
        dimension = self.input_dim if not decoder else self.latent_dim
        for _hidden in hidden_layers:
            if verbose:
                print(
                    f"Adding layer with input dimension: {dimension}, output dimension: {_hidden}")
            layers.extend([
                nn.Linear(dimension, _hidden, device=self.device),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            dimension = _hidden
        if verbose:
            print(
                f"Adding final layer with input dimension: {dimension}, output dimension: {self.latent_dim if not decoder else self.input_dim}")
        layers.append(
            nn.Linear(dimension, self.latent_dim if not decoder else self.input_dim, device=self.device))
        return nn.Sequential(*layers).to(self.device)

    def flatten(self, radiant_picks: list[int], dire_picks: list[int],
                radiant_bans: list[int], dire_bans: list[int],
                radiant_hero_roles: list[list[int]], dire_hero_roles: list[list[int]],
                radiant_stats: list[list[float]], dire_stats: list[list[float]],
                verbose: bool = False
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

        def pad_roles(roles):
            return roles + [0] * (self.n_roles - len(roles))

        radiant_hero_roles_padded = [pad_roles(roles)
                                     for roles in radiant_hero_roles]
        dire_hero_roles_padded = [pad_roles(roles)
                                  for roles in dire_hero_roles]

        radiant_hero_roles_tensor = torch.tensor(
            radiant_hero_roles_padded, dtype=torch.long, device=self.device)
        dire_hero_roles_tensor = torch.tensor(
            dire_hero_roles_padded, dtype=torch.long, device=self.device)

        radiant_hero_roles_feat: torch.Tensor = self.hero_role_embedding(
            radiant_hero_roles_tensor)
        dire_hero_roles_feat: torch.Tensor = self.hero_role_embedding(
            dire_hero_roles_tensor)

        # Stats são normalizados e convertidos em tensores
        radiant_stats_feat: list[torch.Tensor] = [torch.tensor(
            stats, device=self.device) for stats in radiant_stats]
        dire_stats_feat: list[torch.Tensor] = [torch.tensor(
            stats, device=self.device) for stats in dire_stats]

        if (verbose):
            print(
                f"Radiant picks shape: {radiant_picks_feat.shape}, Dire picks shape: {dire_picks_feat.shape}")
            print(
                f"Radiant bans shape: {radiant_bans_feat.shape}, Dire bans shape: {dire_bans_feat.shape}")

            for stats in radiant_hero_roles_feat:
                print(f"Radiant Hero roles tensor shape: {stats.shape}")
            for stats in radiant_stats_feat:
                print(f"Radiant Stats tensor shape: {stats.shape}")
            for stats in dire_hero_roles_feat:
                print(f"Dire Hero roles tensor shape: {stats.shape}")
            for stats in dire_stats_feat:
                print(f"Dire Stats tensor shape: {stats.shape}")

        # Flatten nested lists of tensors for stats and hero roles
        flat = torch.cat([
            radiant_picks_feat.view(-1),
            dire_picks_feat.view(-1),
            radiant_bans_feat.view(-1),
            dire_bans_feat.view(-1),
            radiant_hero_roles_feat.view(-1),
            dire_hero_roles_feat.view(-1),
            torch.cat(radiant_stats_feat).view(-1),
            torch.cat(dire_stats_feat).view(-1)
        ]).unsqueeze(0)

        if (verbose):
            print(f"Flattened tensor shape: {flat.shape}")

        return flat

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
                   radiant_stats: list[list[float]], dire_stats: list[list[float]],
                   verbose: bool = False
                   ) -> float:
        self.train()
        self.optimizer.zero_grad()
        original = self.flatten(radiant_picks, dire_picks,
                                radiant_bans, dire_bans,
                                radiant_hero_roles, dire_hero_roles,
                                radiant_stats, dire_stats,
                                verbose)
        latent, reconstructed = self.forward(original)
        loss = self.loss(original, reconstructed)
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(loss.item())
        return loss.item()

    def train_data(self, training_df: pl.DataFrame, epochs: int = 10, verbose=False) -> None:
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
                dire_stats: list[list[float]] = row['dire_stats']

                min_stats = np.array(row['min_stats'])
                max_stats = np.array(row['max_stats'])

                diff_stats = max_stats - min_stats
                # Evita divisão por zero
                diff_stats = np.where(diff_stats == 0, 1, diff_stats)

                radiant_stats = [
                    ((np.array(stats) - min_stats) / diff_stats).tolist() for stats in radiant_stats]
                dire_stats = [((np.array(stats) - min_stats) /
                               diff_stats).tolist() for stats in dire_stats]

                loss = self.train_step(
                    radiant_picks, dire_picks, radiant_bans, dire_bans,
                    radiant_hero_roles, dire_hero_roles,
                    radiant_stats, dire_stats,
                    verbose
                )
                total_loss += loss
            avg_loss = total_loss / len(training_df)
            if (verbose):
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
            elif (epoch % 10 == 0):
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
