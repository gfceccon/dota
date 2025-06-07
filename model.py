import math
import time
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
        if(verbose):
            print(f"Input dimension: {self.input_dim}")

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
            layers.extend([
                nn.Linear(dimension, _hidden, device=self.device),
                nn.ReLU(),
                nn.Dropout(self.dropout),
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

        # Stats são convertidos em tensores
        radiant_stats_feat: list[torch.Tensor] = [torch.tensor(
            stats, device=self.device) for stats in radiant_stats]
        dire_stats_feat: list[torch.Tensor] = [torch.tensor(
            stats, device=self.device) for stats in dire_stats]


        # Debug: checa NaN/Inf em cada componente antes do flatten
        def check_nan_inf(tensor, name):
            if torch.isnan(tensor).any():
                print(f"[DEBUG] {name} contém NaN")
            if torch.isinf(tensor).any():
                print(f"[DEBUG] {name} contém Inf")

        check_nan_inf(radiant_picks_feat, 'radiant_picks_feat')
        check_nan_inf(dire_picks_feat, 'dire_picks_feat')
        check_nan_inf(radiant_bans_feat, 'radiant_bans_feat')
        check_nan_inf(dire_bans_feat, 'dire_bans_feat')
        check_nan_inf(radiant_hero_roles_feat, 'radiant_hero_roles_feat')
        check_nan_inf(dire_hero_roles_feat, 'dire_hero_roles_feat')
        for i, t in enumerate(radiant_stats_feat):
            check_nan_inf(t, f'radiant_stats_feat[{i}]')
        for i, t in enumerate(dire_stats_feat):
            check_nan_inf(t, f'dire_stats_feat[{i}]')

        # Otimização: converte stats para um único tensor 2D antes do flatten
        radiant_stats_tensor = torch.tensor(radiant_stats, device=self.device)
        dire_stats_tensor = torch.tensor(dire_stats, device=self.device)

        # Otimização: usa .reshape(-1) ao invés de .view(-1) para maior robustez
        flat = torch.cat([
            radiant_picks_feat.reshape(-1),
            dire_picks_feat.reshape(-1),
            radiant_bans_feat.reshape(-1),
            dire_bans_feat.reshape(-1),
            radiant_hero_roles_feat.reshape(-1),
            dire_hero_roles_feat.reshape(-1),
            radiant_stats_tensor.reshape(-1),
            dire_stats_tensor.reshape(-1)
        ]).unsqueeze(0)

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
                   ):
        self.train()
        self.optimizer.zero_grad()
        original = self.flatten(radiant_picks, dire_picks,
                                radiant_bans, dire_bans,
                                radiant_hero_roles, dire_hero_roles,
                                radiant_stats, dire_stats,)
        latent, reconstructed = self.forward(original)
        # Debug: checa se há NaN ou Inf nos tensores
        assert not torch.isnan(original).any(), "original contém NaN"
        assert not torch.isinf(original).any(), "original contém Inf"
        assert not torch.isnan(reconstructed).any(), "reconstructed contém NaN"
        assert not torch.isinf(reconstructed).any(), "reconstructed contém Inf"
        loss = self.loss(original, reconstructed)
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(loss.item())
        return loss.item()

    def train_data(self, training_df: pl.DataFrame, validation_df: pl.DataFrame, epochs: int = 10, verbose=False) -> None:
        for epoch in range(epochs):
            total_loss = 0.0
            total_val_loss = 0.0
            timer_start = time.time()
            train_step_timer_start = 0
            train_step_timer_end = 0
            # Treinamento
            for row in training_df.iter_rows(named=True):
                radiant_picks = row['radiant_picks']
                dire_picks = row['dire_picks']

                radiant_bans = row['radiant_bans']
                dire_bans = row['dire_bans']

                radiant_hero_roles = row['radiant_hero_roles']
                dire_hero_roles = row['dire_hero_roles']

                radiant_stats: list[list[float]] = row['radiant_stats_normalized']
                dire_stats: list[list[float]] = row['dire_stats_normalized']

                train_step_timer_start += time.time()
                loss = self.train_step(
                    radiant_picks, dire_picks, radiant_bans, dire_bans,
                    radiant_hero_roles, dire_hero_roles,
                    radiant_stats, dire_stats,
                )
                
                assert math.isnan(loss) == False, "Loss is NaN, check your data and model configuration."
                train_step_timer_end += time.time()
                total_loss += loss
            # Validação
            with torch.no_grad():
                for row in validation_df.iter_rows(named=True):
                    radiant_picks = row['radiant_picks']
                    dire_picks = row['dire_picks']
                    radiant_bans = row['radiant_bans']
                    dire_bans = row['dire_bans']
                    radiant_hero_roles = row['radiant_hero_roles']
                    dire_hero_roles = row['dire_hero_roles']
                    radiant_stats: list[list[float]] = row['radiant_stats_normalized']
                    dire_stats: list[list[float]] = row['dire_stats_normalized']
                    original = self.flatten(
                        radiant_picks, dire_picks, radiant_bans, dire_bans,
                        radiant_hero_roles, dire_hero_roles, radiant_stats, dire_stats)
                    _, reconstructed = self.forward(original)
                    val_loss = self.loss(original, reconstructed).item()
                    total_val_loss += val_loss
            timer_end = time.time()
            avg_loss = total_loss / len(training_df)
            avg_val_loss = total_val_loss / len(validation_df)
            if (verbose):
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                print(f'Time taken: {timer_end - timer_start:.2f} seconds')
                print(f'Train step average time: {(train_step_timer_end - train_step_timer_start) / len(training_df):.4f} seconds')
            elif (epoch % 10 == 0):
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    def save_model(self, path: str):
        """Salva os pesos do modelo e hiperparâmetros essenciais."""
        checkpoint = {
            'state_dict': self.state_dict(),
            'model_args': {
                'hero_pick_embedding_dim': self.hero_embedding_dim,
                'hero_role_embedding_dim': self.hero_role_embedding_dim,
                'n_player_stats': self.n_player_stats,
                'n_roles': self.n_roles,
                'n_heroes': self.n_heroes,
                'n_players': self.n_players,
                'n_bans': self.n_bans,
                'latent_dim': self.latent_dim,
                'hidden_layers': self.hidden_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate
            }
        }
        torch.save(checkpoint, path)
        print(f"Modelo salvo em {path}")

    @classmethod
    def load_model(cls, path: str, map_location=None, **override_args):
        """Carrega o modelo salvo e retorna uma instância pronta para uso."""
        checkpoint = torch.load(path, map_location=map_location)
        model_args = checkpoint['model_args']
        model_args.update(override_args)  # permite sobrescrever argumentos
        model = cls(**model_args)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        print(f"Modelo carregado de {path}")
        return model

    def save_loss_history(self, path: str):
        """Salva o histórico de loss em um arquivo texto ou csv."""
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['loss'])
            for loss in self.loss_history:
                writer.writerow([loss])
        print(f"Histórico de loss salvo em {path}")
