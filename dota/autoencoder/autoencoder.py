from typing import Optional, Any
import numpy as np
import torch.nn as nn
import polars as pl
import torch
import csv
from dota import get_logger


log = get_logger("Dota2")

config = {
    "latent_dim": 2,
    "hero_pick_embedding_dim": 16,
    "hero_role_embedding_dim": 8,
    "creeps_stacked": [18.0, 2.0, 16.0, 13.0, 11.0, 17.0, 4.0, 9.0, 4.0, 5.0],
    "camps_stacked": [7.0, 1.0, 3.0, 5.0, 4.0, 5.0, 2.0, 2.0, 2.0, 2.0],
    "rune_pickups": [1.0, 8.0, 7.0, 5.0, 4.0, 8.0, 12.0, 2.0, 6.0, 10.0],
    "firstblood_claimed": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "towers_killed": [0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 4.0, 0.0, 1.0, 2.0],
    "roshans_killed": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    "stuns": [
      19.366333, 29.667229, 4.8666625, 41.833046, 73.43217, 21.066572,
      59.600925, 0.0, 13.0, 0.0
    ],
    "item_0": [1, 1, 63, 73, 1466, 73, 141, 178, 1, 135],
    "item_1": [229, 127, 263, 102, 180, 96, 63, 108, 534, 141],
    "item_2": [40, 90, 229, 73, 116, 100, 1, 0, 0, 603],
    "item_3": [180, 242, 218, 50, 26, 119, 156, 231, 123, 116],
    "item_4": [218, 116, 247, 0, 108, 73, 116, 273, 108, 236],
    "item_5": [36, 114, 30, 108, 141, 214, 250, 73, 166, 168],
    "backpack_0": [188, 0, 38, 0, 0, 0, 0, 0, 157, 0],
    "backpack_1": [0, 0, 0, 0, 0, 4204, 0, 4204, 29, 63],
    "backpack_2": [0, 0, 73, 0, 0, 38, 0, 38, 41, 0],
}

class Dota2AE(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hero_pick_embedding_dim: int,
        hero_role_embedding_dim: int,
        dict_roles: dict[str, int],
        hero_cols: list[str],
        player_cols: list[str],
        n_heroes: int,
        n_players: int = 5,
        n_bans: int = 7,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        hidden_layers: list[int] = [256, 128, 64, 32],
    ):
        super(Dota2AE, self).__init__()

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
        self.name = f"Dota2AE_{self.input_dim}_{"_".join([str(h) for h in hidden_layers])}_{latent_dim}"
        self.best_filename = f"{self.name}_best.h5"

        # Inicializa as camadas do modelo
        self.encoder = self.create_encoder()
        self.decoder = self.create_encoder(decoder=True)

        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        self.loss = nn.MSELoss()

        # Históricos de loss
        self.train_stopped = 0
        self.loss_history = []
        self.val_loss_history = []
        self.full_loss_history = []
        self.best_val_loss = float('inf')

    def compute_input_dim(self):
        picks_dim = 2 * self.n_players * self.hero_pick_embedding_dim
        bans_dim = 2 * self.n_bans * self.hero_pick_embedding_dim
        stats_dim = 2 * self.n_players * self.n_player_stats
        hero_stats_dim = 2 * self.n_players * self.n_heroes_stats
        hero_role_dim = 2 * self.n_players * self.n_roles * self.hero_role_embedding_dim
        self.input_dim = picks_dim + bans_dim + hero_role_dim + stats_dim + hero_stats_dim
        return self.input_dim

    def create_encoder(self, decoder: bool = False) -> nn.Sequential:
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
            idx_radiant_picks = columns.index('radiant_picks_idx')
            idx_dire_picks = columns.index('dire_picks_idx')
            idx_radiant_bans = columns.index('radiant_bans_idx')
            idx_dire_bans = columns.index('dire_bans_idx')
            idx_radiant_hero_roles = columns.index('radiant_hero_roles')
            idx_dire_hero_roles = columns.index('dire_hero_roles')
            idx_radiant_features = columns.index('radiant_features')
            idx_dire_features = columns.index('dire_features')
            idx_radiant_hero_features = columns.index('radiant_hero_features')
            idx_dire_hero_features = columns.index('dire_hero_features')
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
        ], dim=1).to(self.device)
        return flat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

    def encode(self, data: np.ndarray[Any, Any], batch_size: int, columns: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = self.flatten(data, batch_size, columns)
        latent, reconstructed = self.forward(tensor)
        return latent, reconstructed

    def train_data(self,
                   training_df: pl.DataFrame, validation_df: pl.DataFrame,
                   epochs: int = 10, batch_size=32, early_stopping: bool = True,
                   patience: int = 10, min_delta: float = 1e-4) -> None:
        epochs_no_improve = 0
        best_state = None
        self.train_stopped = epochs
        self.best_val_loss = float('inf')
        log.info(f"Iniciando treinamento do modelo com {epochs} épocas")
        training_log = "Treinamento"
        log.timer_start(training_log)
        for epoch in range(epochs):
            log.checkpoint(training_log, f"Epoch {epoch + 1}/{epochs}")
            total_loss = 0.0
            total_val_loss = 0.0

            self.train()
            for batch in training_df.iter_slices(batch_size):
                batch_np = batch.to_numpy()
                data = self.flatten(batch_np, min(
                    batch_size, batch_np.shape[0]), training_df.columns)

                self.optimizer.zero_grad()
                latent, reconstructed = self.forward(data)
                loss = self.loss(data, reconstructed)
                loss.backward()
                self.optimizer.step()
                self.full_loss_history.append(loss.item())

                total_loss += loss.item()
                
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

            avg_loss = total_loss / min(count, 1)
            avg_val_loss = total_val_loss / min(count, 1)

            self.loss_history.append(avg_loss)
            self.val_loss_history.append(avg_val_loss)

            if early_stopping:
                if avg_val_loss < self.best_val_loss - min_delta:
                    self.best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    best_state = self.state_dict()
                    
                    self.save_model(self.best_filename,
                                    verbose=self.verbose, silent=self.silent)
                    if self.verbose and not self.silent:
                        log.info(
                            f"Melhor modelo salvo em {self.best_filename} (Val Loss: {self.best_val_loss:.4f})")
                else:
                    epochs_no_improve += 1
                    if self.verbose and not self.silent:
                        log.info(
                            f"Nenhuma melhora na validação por {epochs_no_improve} épocas.")
                if epochs_no_improve >= patience:
                    self.train_stopped = epoch + 1
                    log.info(f"Early stopping ativado após {epoch+1} épocas.")
                    log.info(
                        f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                    log.timer_end(training_log)
                    break
            log.checkpoint(training_log, f"Final da Epoch {epoch + 1}/{epochs}")
        # Carrega o melhor modelo ao final do treinamento
        if early_stopping and best_state is not None:
            with torch.serialization.safe_globals([pl.series.series.Series]):
                self.load_state_dict(torch.load(
                    self.best_filename)['state_dict'])
            log.timer_end(training_log)

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

    def save_model(self, path: Optional[str] = None, verbose: bool = False, silent: bool = False):
        self.silent = silent
        self.verbose = verbose
        if path is None:
            path = f"./best/{self.base_filename}.h5"
        checkpoint = {
            'model_args': {
                'name': self.name,
                'hero_pick_embedding_dim': self.hero_pick_embedding_dim,
                'hero_role_embedding_dim': self.hero_role_embedding_dim,
                'dict_roles': self.dict_roles,
                'hero_cols': self.hero_columns,
                'player_cols': self.player_columns,
                'n_heroes': self.n_heroes,
                'n_players': self.n_players,
                'n_bans': self.n_bans,
                'latent_dim': self.latent_dim,
                'hidden_layers': self.hidden_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate
            },
            'state_dict': self.state_dict(),
            'full_loss_history': self.full_loss_history,
            'best_val_loss': self.best_val_loss,
            'train_stopped': self.train_stopped
        }
        torch.save(checkpoint, path)
        if (self.verbose and not self.silent):
            log.info(f"Modelo salvo em {path}")

    @classmethod
    def load_model(cls, path: str, **override_args):
        location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.serialization.safe_globals([pl.series.series.Series]):
            checkpoint = torch.load(path, map_location=location)
            model_args = checkpoint['model_args']
            model_args.update(override_args)
            model = cls(**model_args)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            log.info(f"Modelo carregado de {path}")
            return model

    def save_loss_history(self, path: Optional[str] = None, silent: bool = False):
        self.silent = silent
        if path is None:
            path = f"./best/{self.base_filename}_loss.csv"
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['loss', 'eval_loss'])
            for loss, eval in zip(self.loss_history, self.val_loss_history):
                writer.writerow([loss, eval])
        if not self.silent:
            log.info(f"Histórico de loss salvo em {path}")
