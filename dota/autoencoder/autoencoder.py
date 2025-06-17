from typing import Optional, Any, cast
import torch.nn as nn
from torch import FloatTensor, Tensor, LongTensor, IntTensor
import numpy as np
import pandas as pd
import torch
from dota.logger import get_logger


log = get_logger("Dota2")


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

        embeddings: list[tuple[int, int]],
        embeddings_config: dict[str, bool],
    ):
        super(Dota2AE, self).__init__()
        self.name = name
        # Configura o device para GPU se disponível, caso contrário CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Camadas de embedding para heróis, bans e estatísticas
        self.embeddings = {
            name: nn.Embedding(emb[0], emb[1], device=self.device)
            for name, emb in zip(embeddings_config.keys(), embeddings)
        }

        # Dimensões e hiperparâmetros do modelo
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.learning_rate = lr
        self.dropout = dropout

        self.name = f"Dota2AE_{name}"
        self.best_filename = f"Dota2AE_{name}_best.h5"

        # Inicializa as camadas do modelo
        self.encoder, self.decoder = self.create_ae(
            input_dim, latent_dim, encoder_layers, decoder_layers)

        self.optimizer = torch.optim.Adam(self.parameters(), lr)
        self.loss = nn.MSELoss()

        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.early_stopping = early_stopping

        # Históricos de loss
        self.train_stopped = 0
        self.loss_history = []
        self.val_loss_history = []
        self.full_loss_history = []
        self.best_val_loss = float('inf')

    def create_ae(self, input_dim: int, latent_dim, encoder_layers: list[int], decoder_layers: list[int]) -> tuple[nn.Sequential, nn.Sequential]:
        _layers = []
        dim = input_dim
        for _hidden in encoder_layers:
            _layers.extend([
                nn.Linear(dim, _hidden, device=self.device),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            dim = _hidden

        _layers.append(nn.Linear(dim, latent_dim, device=self.device))
        _layers.append(nn.ReLU())

        encoder = nn.Sequential(*_layers).to(self.device)
        _layers = []

        dim = latent_dim
        for _hidden in decoder_layers:
            _layers.extend([
                nn.Linear(dim, _hidden, device=self.device),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            dim = _hidden

        _layers.append(nn.Linear(dim, input_dim, device=self.device))
        _layers.append(nn.ReLU())
        _layers.append(nn.Sigmoid())

        decoder = nn.Sequential(*_layers).to(self.device)

        return encoder, decoder

    def flatten(self, data: np.ndarray[Any, Any]) -> torch.Tensor:

        flat = torch.cat([
        ], dim=1).to(self.device)
        return flat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

    def encode(self, data: np.ndarray[Any, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = self.flatten(data)
        latent, reconstructed = self.forward(tensor)
        return latent, reconstructed

    def train_data(self, training_df_path: str, validation_df_path: str, columns: dict[str, bool], emb_columns: dict[str, bool]) -> None:
        def pad_sequences(sequences, dtype, value=0):
            """
            Faz padding em uma lista de arrays/listas para que todos tenham o mesmo tamanho.
            """
            max_len = max(len(seq) for seq in sequences)
            return np.array([
                np.pad(seq, (0, max_len - len(seq)),
                       mode='constant', constant_values=value)
                for seq in sequences
            ], dtype=dtype)

        epochs_no_improve = 0
        best_state = None
        self.train_stopped = self.epochs
        self.best_val_loss = float('inf')
        log.info(f"Iniciando treinamento do modelo com {self.epochs} épocas")
        training_log = "Treinamento"
        log.timer_start(training_log)
        for epoch in range(self.epochs):
            log.checkpoint(training_log, f"Epoch {epoch + 1}/{self.epochs}")
            total_loss = 0.0
            total_val_loss = 0.0

            self.train()
            for _data in pd.read_json(training_df_path, orient='records', lines=True, chunksize=self.batch_size):
                _data = cast(pd.DataFrame, _data)
                tensor_array = []
                for col in columns:
                    log.info(f"Processando coluna: {col}")
                    if columns[col]:
                        tensor_array.append(torch.tensor(pad_sequences(
                            _data[col].values, dtype=np.float32), device=self.device))
                for col in emb_columns:
                    emb = self.embeddings[col]
                    if isinstance(emb, nn.Embedding) and emb_columns[col]:
                        log.info(f"Processando coluna de embedding: {col}")
                        embedded: Tensor = emb(torch.tensor(pad_sequences(_data[col].values, dtype=np.int32),device=self.device))
                        tensor_array.append(embedded)

                data = torch.cat(tensor_array, dim=0).to(self.device)
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
                for batch_eval in pd.read_csv(validation_df_path, chunksize=self.batch_size):
                    _, reconstructed = self.forward(batch_eval)
                    val_loss = self.loss(batch_eval, reconstructed).item()
                    total_val_loss += val_loss
                    count += 1

            avg_loss = total_loss / min(count, 1)
            avg_val_loss = total_val_loss / min(count, 1)

            self.loss_history.append(avg_loss)
            self.val_loss_history.append(avg_val_loss)

            if self.early_stopping:
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    best_state = self.state_dict()

                    # self.save_model(self.best_filename)
                    if self.verbose and not self.silent:
                        log.info(
                            f"Melhor modelo salvo em {self.best_filename} (Val Loss: {self.best_val_loss:.4f})")
                else:
                    epochs_no_improve += 1
                    if self.verbose and not self.silent:
                        log.info(
                            f"Nenhuma melhora na validação por {epochs_no_improve} épocas.")
                if epochs_no_improve >= self.patience:
                    self.train_stopped = epoch + 1
                    log.info(f"Early stopping ativado após {epoch+1} épocas.")
                    log.info(
                        f'Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                    log.timer_end(training_log)
                    break
            log.checkpoint(
                training_log, f"Final da Epoch {epoch + 1}/{self.epochs}")
            # Carrega o melhor estado do modelo se houver
            if best_state is not None:
                self.load_state_dict(best_state)
            log.timer_end(training_log)

#     def test_model(self, test_df: pl.DataFrame, batch_size: int = 32, threshold=0.01) -> tuple[float, float, float, float]:
#         self.eval()
#         correct = 0
#         total = 0
#         total_mse = 0.0
#         min_mse = float('inf')
#         max_mse = float('-inf')
#         with torch.no_grad():
#             for batch in test_df.iter_slices(batch_size):
#                 batch_np = batch.to_numpy()
#                 original = self.flatten(batch_np, min(
#                     batch_size, batch_np.shape[0]), columns=test_df.columns)
#                 latent, reconstructed = self.forward(original)
#                 mse = self.loss(original, reconstructed).item()
#                 total_mse += float(mse)
#                 min_mse = min(min_mse, mse)
#                 max_mse = max(max_mse, mse)
#                 if mse < threshold:
#                     correct += 1
#                 total += 1
#         accuracy = correct / total if total > 0 else 1
#         avg_mse = total_mse / total if total > 0 else 1
#         return accuracy, avg_mse, min_mse, max_mse

    def save_model(self):
        path = f"./best/{self.name}.h5"
        checkpoint = {
            'model_args': {
                'name': self.name,
                'lr': self.learning_rate,
                'dropout': self.dropout,
                'early_stopping': self.early_stopping,
                'epochs': self.epochs,
                'patience': self.patience,
                'batch_size': self.batch_size,
                'input_dim': self.input_dim,
                'latent_dim': self.latent_dim,
                'encoder_layers': self.encoder_layers,
                'decoder_layers': self.decoder_layers,
                'embeddings': [(emb.weight.shape[0], emb.weight.shape[1]) for emb in self.embeddings.values()],
                'embeddings_config': {k: v for k, v in self.embeddings.items()},
            },
            'state_dict': self.state_dict(),
        }
        torch.save(checkpoint, path)
        log.info(f"Modelo salvo em {path}")

#     @classmethod
#     def load_model(cls, path: str, **override_args):
#         location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         with torch.serialization.safe_globals([pl.series.series.Series]):
#             checkpoint = torch.load(path, map_location=location)
#             model_args = checkpoint['model_args']
#             model_args.update(override_args)
#             model = cls(**model_args)
#             model.load_state_dict(checkpoint['state_dict'])
#             model.eval()
#             log.info(f"Modelo carregado de {path}")
#             return model

#     def save_loss_history(self, path: Optional[str] = None, silent: bool = False):
#         self.silent = silent
#         if path is None:
#             path = f"./best/{self.base_filename}_loss.csv"
#         with open(path, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['loss', 'eval_loss'])
#             for loss, eval in zip(self.loss_history, self.val_loss_history):
#                 writer.writerow([loss, eval])
#         if not self.silent:
#             log.info(f"Histórico de loss salvo em {path}")
