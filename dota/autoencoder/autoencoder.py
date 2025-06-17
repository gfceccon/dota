import csv
from typing import Optional, Any, cast
import torch.nn as nn
from torch import FloatTensor, Tensor, LongTensor, IntTensor
import numpy as np
import pandas as pd
import torch
from dota.logger import get_logger

log = get_logger("Dota2AE", log_file="autoencoder.log")


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

        # Configura o device para GPU se disponível, caso contrário CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

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

        # Inicializa as camadas do modelo
        self.encoder, self.decoder = self.create_ae(
            input_dim, latent_dim, encoder_layers, decoder_layers)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr)
        self.loss = nn.MSELoss()

        # Históricos de loss
        self.train_stopped = 0
        self.loss_val_history = []
        self.loss_history = []
        self.best_val_loss = float('inf')

    def create_ae(self, input_dim: int, latent_dim, encoder_layers: list[int], decoder_layers: list[int]) -> tuple[nn.Sequential, nn.Sequential]:
        # Encoder
        _layers = []
        dim = input_dim
        for _hidden in encoder_layers:
            _layers.extend([
                nn.Linear(dim, _hidden, device=self.device),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            dim = _hidden

        # Última camada do encoder para o espaço latente
        _layers.append(nn.Linear(dim, latent_dim, device=self.device))
        _layers.append(nn.ReLU())
        encoder = nn.Sequential(*_layers).to(self.device)

        # Decoder
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
        decoder = nn.Sequential(*_layers).to(self.device)

        return encoder, decoder

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

    def encode(self, data: np.ndarray[Any, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = torch.tensor(data, device=self.device, dtype=torch.float32)
        latent, reconstructed = self.forward(tensor)
        return latent, reconstructed

    def pad_sequences(self, sequences, dtype, value=0):
        max_len = max(len(seq) for seq in sequences)
        return np.array([
            np.pad(seq, (0, max_len - len(seq)),
                   mode='constant', constant_values=value)
            for seq in sequences
        ], dtype=dtype)

    def flatten_tensor(self, batch, batch_size, columns, radiant, dire, emb_columns) -> Tensor:
        tensor_array = []
        # Processar colunas numéricas
        radiant_stack = np.vstack(batch[radiant].values)
        dire_stack = np.vstack(batch[dire].values)
        log.info(f"Processando da coluna: {radiant} e {dire}")
        log.info(
            f"Dimensão do stack: {radiant_stack.shape} e {dire_stack.shape}")


        tensor = torch.tensor(radiant_stack, device=self.device)
        tensor = tensor.view(batch_size, -1)
        tensor_array.append(tensor)

        tensor = torch.tensor(dire_stack, device=self.device)
        tensor = tensor.view(batch_size, -1)
        tensor_array.append(tensor)

        # Processar embeddings
        for col in emb_columns:
            embed = self.embeddings[col]
            if isinstance(embed, nn.Embedding) and emb_columns[col]:
                col_padded = np.vstack(batch[col].values, dtype=np.int32)
                log.info(
                    f"Processando do embedding: {col}",
                    f"Dimensão do embedding: {batch[col].values.shape}",
                    f"Dimensão após padding: {col_padded.shape}")
                ids = torch.tensor(col_padded, device=self.device)
                embedded = embed(ids)
                embedded = embedded.view(batch_size, -1)
                tensor_array.append(embedded)
        # Concatenar todos os tensores ao longo da dimensão 1 (colunas)
        data = torch.cat(tensor_array, dim=1).to(self.device)
        return data

    def train_data(self, training_df_path: str, validation_df_path: str, columns, radiant, dire, emb_columns: dict[str, bool]) -> None:
        epochs_no_improve = 0
        best_state = None
        self.train_stopped = self.epochs
        self.best_val_loss = float('inf')
        log.info(f"Iniciando treinamento do modelo com {self.epochs} épocas")
        training_log = "Treinamento"
        log.timer_start(training_log)
        for epoch in range(self.epochs):
            total_loss = 0.0
            total_val_loss = 0.0
            size_loss = 0
            size_val_loss = 0
            self.train()
            for _data in pd.read_json(training_df_path, orient='records', lines=True, chunksize=self.batch_size):
                batch_data = cast(pd.DataFrame, _data)
                batch_size = len(batch_data)
                input_sum = 0
                data = self.flatten_tensor(batch_data, batch_size, columns, radiant, dire, emb_columns)
                
                input_sum = data.shape[1]
                
                if (input_sum != self.input_dim):
                    log.critical(
                        f"Dimensão de entrada {input_sum} não corresponde à dimensão esperada {self.input_dim}. Verifique as colunas e embeddings.")
                    raise ValueError(
                        f"Dimensão de entrada {input_sum} não corresponde à dimensão esperada {self.input_dim}. Verifique as colunas e embeddings.")
                log.debug(f"Dimensão total de entrada: {input_sum}")
                self.optimizer.zero_grad()
                # Passar pelo modelo
                latent, reconstructed = self.forward(data)
                loss = self.loss(data, reconstructed)
                # Calcular a perda
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                size_loss += batch_size

            avg_loss = total_loss / size_loss
            self.loss_history.append(avg_loss)
            
            log.checkpoint(
                training_log, f"Treinamento da Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
            
            log.info(
                f"Validando modelo após {epoch + 1} épocas de treinamento")
            
            self.eval()
            with torch.no_grad():
                for val_batch_data in pd.read_json(validation_df_path, orient='records', lines=True, chunksize=self.batch_size):
                    val_data = cast(pd.DataFrame, val_batch_data)
                    val_batch_size = len(val_batch_data)
                    size_val_loss += val_batch_size
                    val_data = self.flatten_tensor(val_data, val_batch_size, columns, radiant, dire, emb_columns)
                    
                    # Passar pelo modelo
                    _, reconstructed = self.forward(val_data)
                    val_loss = self.loss(val_data, reconstructed).item()
                    total_val_loss += val_loss

            avg_val_loss = total_val_loss / size_val_loss
            self.loss_val_history.append(avg_val_loss)

            if self.early_stopping:
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    # best_state = self.state_dict()
                    # self.save_model()
                else:
                    epochs_no_improve += 1
                    log.info(
                        f"Nenhuma melhora na validação por {epochs_no_improve} épocas.")
                    log.info(f"Loss {self.best_val_loss:.4f} -> {avg_val_loss:.4f}")
                if epochs_no_improve >= self.patience:
                    self.train_stopped = epoch + 1
                    log.info(f"Early stopping ativado após {epoch+1} épocas.")
                    log.info(
                        f'Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                    log.timer_end(training_log)
                    break
            log.checkpoint(
                training_log, f"Validação da Epoch {epoch + 1}/{self.epochs}")
            
        # Carrega o melhor estado do modelo se houver
        if best_state is not None:
            self.load_state_dict(best_state)
        log.timer_end(training_log)
        log.info(
            f'Epoch {self.train_stopped}/{self.epochs}, Loss: {self.best_val_loss:.4f}')

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

    @classmethod
    def load_model(cls, path: str, **override_args):
        location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=location)
        model_args = checkpoint['model_args']
        model_args.update(override_args)
        model = cls(**model_args)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        log.info(f"Modelo carregado de {path}")
        return model

    def save_loss_history(self, ):
        with open(self.loss_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['loss', 'eval_loss'])
            for loss, eval in zip(self.loss_history, self.loss_val_history):
                writer.writerow([loss, eval])
        if not getattr(self, 'silent', False):
            log.info(f"Histórico de loss salvo em {self.loss_path}")
