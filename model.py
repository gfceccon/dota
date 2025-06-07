from typing import Any
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
        n_player_stats: int,
        dict_roles: dict[str, int],
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
        self.n_player_stats = n_player_stats
        self.n_players = n_players
        self.n_bans = n_bans
        self.n_picks = n_players + n_bans
        
        # Dicionário de roles
        self.dict_roles = dict_roles
        self.n_roles = len(dict_roles)

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
        self.hero_embedding_dim = hero_pick_embedding_dim
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
        self.avg_eval_history = []

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
                nn.Linear(dimension, _hidden, device=self.device), nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            dimension = _hidden
        layers.append(
            nn.Linear(dimension, self.latent_dim if not decoder else self.input_dim, device=self.device))
        return nn.Sequential(*layers).to(self.device)

    def flatten(self, data: dict[str, Any]) -> torch.Tensor:
        
        radiant_picks = data['radiant_picks']
        dire_picks = data['dire_picks']

        radiant_bans = data['radiant_bans']
        dire_bans = data['dire_bans']

        radiant_hero_roles = data['radiant_hero_roles']
        dire_hero_roles = data['dire_hero_roles']

        radiant_stats: list[list[float]
                            ] = data['radiant_stats_normalized']
        dire_stats: list[list[float]] = data['dire_stats_normalized']

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

    def encode(self, data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = self.flatten(data)
        latent, reconstructed = self.forward(tensor)
        return latent, reconstructed

    def train_data(self, training_df: pl.DataFrame, validation_df: pl.DataFrame, epochs: int = 10, verbose=False) -> None:
        for epoch in range(epochs):
            total_loss = 0.0
            total_val_loss = 0.0
            timer_start = time.time()
            
            self.train()
            # Treinamento do modelo
            for row in training_df.iter_rows(named=True):
                data = self.flatten(row)
                
                self.optimizer.zero_grad()
                latent, reconstructed = self.forward(data)
                loss = self.loss(data, reconstructed)
                loss.backward()
                self.optimizer.step()
                self.loss_history.append(loss.item())
                
                total_loss += loss.item()
            # Validação do modelo
            self.eval()
            with torch.no_grad():
                for row in validation_df.iter_rows(named=True):
                    original = self.flatten(row)
                    _, reconstructed = self.forward(original)
                    val_loss = self.loss(original, reconstructed).item()
                    total_val_loss += val_loss
                    
            # Fim da época   
            timer_end = time.time()
            
            # Calcula a média de loss para a época
            avg_loss = total_loss / len(training_df)
            avg_val_loss = total_val_loss / len(validation_df)
            
            # Armazena os resultados médios
            self.avg_history.append(avg_loss)
            self.avg_eval_history.append(avg_val_loss)
            
            # Exibe os resultados
            if (verbose):
                print(
                    f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                print(f'Time taken: {timer_end - timer_start:.2f} seconds')
            elif (epoch % 10 == 0):
                print(
                    f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    def test_model(self, test_df: pl.DataFrame) -> float:
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for row in test_df.iter_rows(named=True):
                original = self.flatten(row)
                latent, reconstructed = self.forward(original)
                # Supondo que a tarefa é reconstrução, a acurácia pode ser definida como
                # quão próximo o reconstruído está do original (ex: erro médio quadrático abaixo de um threshold)
                mse = self.loss(original, reconstructed).item()
                threshold = 0.01  # Defina um threshold apropriado para seu caso
                if mse < threshold:
                    correct += 1
                total += 1
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def save_model(self, path: str):
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
            for loss, eval in zip(self.avg_history, self.avg_eval_history):
                writer.writerow([loss, eval])
        print(f"Histórico de loss salvo em {path}")
