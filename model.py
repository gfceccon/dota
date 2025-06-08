import math
from typing import Any
import numpy as np
import torch.nn as nn
import polars as pl
import time
import torch
import csv


class Dota2Autoencoder(nn.Module):
    def __init__(
        self,
        hero_pick_embedding_dim: int,
        hero_role_embedding_dim: int,
        dict_roles: dict[str, int],
        hero_cols: list[str],
        player_cols: list[str],
        match_cols: list[str],
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

        # Número de heróis, estatísticas, jogadores, picks e bans
        self.n_heroes = n_heroes
        self.n_players = n_players
        self.n_bans = n_bans
        self.n_picks = n_players + n_bans

        # Dicionário de roles
        self.dict_roles = dict_roles
        self.n_roles = len(dict_roles)
        self.player_columns = player_cols
        self.n_player_stats = len(player_cols)
        self.hero_columns = hero_cols
        self.n_heroes_stats = len(hero_cols)
        self.match_columns = match_cols
        self.n_match_columns = len(match_cols)

        # Camadas de embedding para heróis, bans e estatísticas
        self.hero_pick_embedding = nn.Embedding(
            n_heroes, hero_pick_embedding_dim, device=self.device)
        self.hero_role_embedding = nn.Embedding(
            2, hero_role_embedding_dim, device=self.device)

        # Dimensões e hiperparâmetros do modelo
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.dropout = dropout

        # Dimensões de embedding para picks e roles
        self.hero_pick_embedding_dim = hero_pick_embedding_dim
        self.hero_role_embedding_dim = hero_role_embedding_dim

        # Calcula a dimensão de entrada do modelo, para cada time
        self.input_dim = self.compute_input_dim()
        if (verbose):
            print(f"Input dimension: {self.input_dim}")

        # Inicializa as camadas do modelo
        self.encoder = self.create_encoder(verbose=verbose)
        self.decoder = self.create_encoder(decoder=True, verbose=verbose)

        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        self.loss = nn.MSELoss()

        # Históricos de loss
        self.loss_history = []
        self.avg_history = []
        self.avg_val_history = []
        self.epoch_stop = 0

    def compute_input_dim(self):
        picks_dim = 2 * self.n_players * self.hero_pick_embedding_dim
        bans_dim = 2 * self.n_bans * self.hero_pick_embedding_dim
        stats_dim = 2 * self.n_players * self.n_player_stats
        hero_dim = 2 * self.n_players * self.n_heroes_stats
        hero_role_dim = 2 * self.n_players * self.n_roles * self.hero_role_embedding_dim
        match_dim = self.n_match_columns
        self.input_dim = picks_dim + bans_dim + \
            stats_dim + hero_role_dim + hero_dim + match_dim
        return self.input_dim

    def create_encoder(self, decoder: bool = False, verbose: bool = False) -> nn.Sequential:
        layers = []
        hidden_layers = self.hidden_layers if not decoder else reversed(
            self.hidden_layers)
        dimension = self.input_dim if not decoder else self.latent_dim
        for _hidden in hidden_layers:
            layers.extend([
                nn.Linear(dimension, _hidden, device=self.device), nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            dimension = _hidden
        layers.append(
            nn.Linear(dimension, self.latent_dim if not decoder else self.input_dim, device=self.device))
        return nn.Sequential(*layers).to(self.device)

    def flatten(self, data: np.ndarray[Any, Any], batch_size: int, columns: list[str]) -> torch.Tensor:
        try:
            idx_radiant_picks = columns.index('radiant_picks')
            idx_dire_picks = columns.index('dire_picks')
            idx_radiant_bans = columns.index('radiant_bans')
            idx_dire_bans = columns.index('dire_bans')
            idx_radiant_hero_roles = columns.index('radiant_hero_roles')
            idx_dire_hero_roles = columns.index('dire_hero_roles')
            idx_radiant_features = columns.index('radiant_features')
            idx_dire_features = columns.index('dire_features')
            idx_radiant_hero_features = columns.index('radiant_hero_features')
            idx_dire_hero_features = columns.index('dire_hero_features')
            idx_duration = columns.index('match_duration_normalized')
            idx_radiant_winner = columns.index('match_winner')
        except ValueError:
            raise KeyError(f"Column not found in columns list.")

        radiant_picks = np.stack(data[:, idx_radiant_picks].tolist())
        dire_picks = np.stack(data[:, idx_dire_picks].tolist())

        radiant_bans = np.stack(data[:, idx_radiant_bans].tolist())
        dire_bans = np.stack(data[:, idx_dire_bans].tolist())

        radiant_hero_roles = np.stack(
            [np.array(d.tolist()) for d in data[:, idx_radiant_hero_roles]])
        dire_hero_roles = np.stack([np.array(d.tolist())
                                   for d in data[:, idx_dire_hero_roles]])

        match_duration = np.stack(data[:, idx_duration].tolist())
        match_radiant_win = np.stack(data[:, idx_radiant_winner].tolist())

        radiant_stats = np.stack(data[:, idx_radiant_features].tolist())
        dire_stats = np.stack(data[:, idx_dire_features].tolist())

        radiant_hero_stats = np.stack(
            data[:, idx_radiant_hero_features].tolist())
        dire_hero_stats = np.stack(data[:, idx_dire_hero_features].tolist())

        # Picks e bans
        radiant_picks_feat: torch.Tensor = self.hero_pick_embedding(
            torch.tensor(radiant_picks, device=self.device))
        radiant_bans_feat: torch.Tensor = self.hero_pick_embedding(
            torch.tensor(radiant_bans, device=self.device))
        dire_picks_feat: torch.Tensor = self.hero_pick_embedding(
            torch.tensor(dire_picks, device=self.device))
        dire_bans_feat: torch.Tensor = self.hero_pick_embedding(
            torch.tensor(dire_bans, device=self.device))

        radiant_hero_roles_tensor = torch.tensor(
            radiant_hero_roles, dtype=torch.long, device=self.device)
        dire_hero_roles_tensor = torch.tensor(
            dire_hero_roles, dtype=torch.long, device=self.device)

        radiant_hero_roles_feat: torch.Tensor = self.hero_role_embedding(
            radiant_hero_roles_tensor)
        dire_hero_roles_feat: torch.Tensor = self.hero_role_embedding(
            dire_hero_roles_tensor)

        radiant_stats_tensor = torch.tensor(
            radiant_stats, device=self.device, dtype=torch.float32)
        dire_stats_tensor = torch.tensor(
            dire_stats, device=self.device, dtype=torch.float32)
        radiant_hero_stats_tensor = torch.tensor(
            radiant_hero_stats, device=self.device, dtype=torch.float32)
        dire_hero_stats_tensor = torch.tensor(
            dire_hero_stats, device=self.device, dtype=torch.float32)
        match_duration_tensor = torch.tensor(
            match_duration, device=self.device, dtype=torch.float32)
        match_radiant_win_tensor = torch.tensor(
            match_radiant_win, device=self.device, dtype=torch.float32)

        flat = torch.cat([
            radiant_picks_feat.reshape(batch_size, -1),
            dire_picks_feat.reshape(batch_size, -1),
            radiant_bans_feat.reshape(batch_size, -1),
            dire_bans_feat.reshape(batch_size, -1),
            radiant_hero_roles_feat.reshape(batch_size, -1),
            dire_hero_roles_feat.reshape(batch_size, -1),
            radiant_stats_tensor.reshape(batch_size, -1),
            dire_stats_tensor.reshape(batch_size, -1),
            radiant_hero_stats_tensor.reshape(batch_size, -1),
            dire_hero_stats_tensor.reshape(batch_size, -1),
            match_duration_tensor.reshape(batch_size, -1),
            match_radiant_win_tensor.reshape(batch_size, -1),
        ], dim=1)
        return flat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encoda e decodifica os dados
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

    def encode(self, data: dict[str, Any], batch_size: int, columns: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        _data = np.array(data)
        tensor = self.flatten(_data, batch_size, columns)
        latent, reconstructed = self.forward(tensor)
        return latent, reconstructed

    def train_data(self,
                   training_df: pl.DataFrame, validation_df: pl.DataFrame,
                   epochs: int = 10, batch_size=32, early_stopping: bool = True,
                   patience: int = 10, min_delta: float = 1e-4,
                   best_model_path: str = "best_model.pth",
                   verbose=False,) -> None:
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_state = None
        epoch_stop = epochs
        for epoch in range(epochs):
            total_loss = 0.0
            total_val_loss = 0.0
            timer_start = time.time()

            self.train()
            # Treinamento do modelo
            training_df.columns
            for batch in training_df.iter_slices(batch_size):
                batch_np = batch.to_numpy()
                data = self.flatten(batch_np, min(
                    batch_size, batch_np.shape[0]), training_df.columns)

                self.optimizer.zero_grad()
                latent, reconstructed = self.forward(data)
                loss = self.loss(data, reconstructed)
                loss.backward()
                self.optimizer.step()
                self.loss_history.append(loss.item())

                total_loss += loss.item()
            # Validação do modelo
            self.eval()
            count = 0
            with torch.no_grad():
                for batch_eval in validation_df.iter_slices(batch_size):
                    batch_eval_np = batch_eval.to_numpy()
                    original = self.flatten(batch_eval_np, min(
                        batch_size, batch_eval_np.shape[0]), validation_df.columns)
                    _, reconstructed = self.forward(original)
                    val_loss = self.loss(original, reconstructed).item()
                    total_val_loss += val_loss
                    count += 1

            # Fim da época
            timer_end = time.time()

            # Calcula a média de loss para a época
            avg_loss = total_loss / min(count, 1)
            avg_val_loss = total_val_loss / min(count, 1)

            # Armazena os resultados médios
            self.avg_history.append(avg_loss)
            self.avg_val_history.append(avg_val_loss)

            # Early stopping e salvamento do melhor modelo
            if early_stopping:
                if avg_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    best_state = self.state_dict()
                    # Salva o melhor modelo
                    self.save_model(best_model_path)
                    if verbose:
                        print(
                            f"Melhor modelo salvo em {best_model_path} (Val Loss: {best_val_loss:.4f})")
                else:
                    epochs_no_improve += 1
                    if verbose:
                        print(
                            f"Nenhuma melhora na validação por {epochs_no_improve} épocas.")
                if epochs_no_improve >= patience:
                    epoch_stop = epoch + 1
                    print(f"Early stopping ativado após {epoch+1} épocas.")
                    print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                    break

            # Exibe os resultados
            if (verbose):
                print(
                    f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                print(f'Time taken: {timer_end - timer_start:.2f} seconds')
            elif ((epoch + 1) % 10 == 0 or epoch == 0):
                print(
                    f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        # Carrega o melhor modelo ao final do treinamento
        if early_stopping and best_state is not None:
            self.load_state_dict(torch.load(best_model_path)['state_dict'])
            
        self.epoch_stop = epoch_stop

    def test_model(self, test_df: pl.DataFrame, batch_size: int = 32, threshold=0.01) -> tuple[float, float, float, float]:
        self.eval()
        correct = 0
        total = 0
        total_mse = 0.0
        min_mse = float('inf')
        max_mse = float('-inf')
        with torch.no_grad():
            for batch in test_df.iter_slices(batch_size):
                batch_np = batch.to_numpy()
                original = self.flatten(batch_np, min(
                    batch_size, batch_np.shape[0]), columns=test_df.columns)
                latent, reconstructed = self.forward(original)
                mse = self.loss(original, reconstructed).item()
                total_mse += float(mse)
                min_mse = min(min_mse, mse)
                max_mse = max(max_mse, mse)
                if mse < threshold:
                    correct += 1
                total += 1
        accuracy = correct / total if total > 0 else 1
        avg_mse = total_mse / total if total > 0 else 1
        return accuracy, avg_mse, min_mse, max_mse

    def save_model(self, path: str, verbose: bool = False):
        checkpoint = {
            'model_args': {
                'hero_pick_embedding_dim': self.hero_pick_embedding_dim,
                'hero_role_embedding_dim': self.hero_role_embedding_dim,
                'dict_roles': self.dict_roles,
                'hero_cols': self.hero_columns,
                'player_cols': self.player_columns,
                'match_cols': self.match_columns,
                'n_heroes': self.n_heroes,
                'n_players': self.n_players,
                'n_bans': self.n_bans,
                'latent_dim': self.latent_dim,
                'hidden_layers': self.hidden_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate
            },
            'state_dict': self.state_dict(),
            'loss_history': self.loss_history,
            'avg_history': self.avg_history,
            'avg_eval_history': self.avg_val_history,
            'epoch_stop': self.epoch_stop
        }
        torch.save(checkpoint, path)
        if(verbose):
            print(f"Modelo salvo em {path}")

    @classmethod
    def load_model(cls, path: str, map_location: torch.device, **override_args):
        checkpoint = torch.load(path, map_location=map_location)
        model_args = checkpoint['model_args']
        model_args.update(override_args)
        model = cls(**model_args)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        print(f"Modelo carregado de {path}")
        return model

    def save_loss_history(self, path: str):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['loss', 'eval_loss'])
            for loss, eval in zip(self.avg_history, self.avg_val_history):
                writer.writerow([loss, eval])
        print(f"Histórico de loss salvo em {path}")
