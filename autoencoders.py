import kagglehub
import torch
import torch.nn as nn
import numpy as np
import polars as pl
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from src.files import (
    heroes_file,
)
from src.dataset import get_dataset





class DotaMatchAutoencoder(nn.Module):
    """
    Autoencoder para dados de partidas de Dota 2 com separação entre times.
    
    Processa:
    - IDs de heróis por time (radiant/dire separadamente)
    - IDs de heróis banidos
    - Métricas de performance (kills, deaths, assists, gold/min, xp/min)
    """
    
    def __init__(
        self,
        max_heroes: int = 150,  # Número máximo de heróis baseado em n_heroes do dataset
        embedding_dim: int = 32,
        hidden_dims: List[int] = [256, 128, 64],
        latent_dim: int = 32,
        dropout_rate: float = 0.2
    ):
        super(DotaMatchAutoencoder, self).__init__()
        
        self.max_heroes = max_heroes
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        
        # Embeddings para heróis
        self.hero_embedding = nn.Embedding(max_heroes + 1, embedding_dim, padding_idx=0)
        
        # Embeddings específicos para cada time (opcional)
        self.team_embedding = nn.Embedding(2, embedding_dim)  # 0: Radiant, 1: Dire
        
        # Dimensões das features - SEPARADAS POR TIME
        self.radiant_heroes_dim = 5 * embedding_dim  # 5 heróis Radiant
        self.dire_heroes_dim = 5 * embedding_dim     # 5 heróis Dire
        self.team_embeddings_dim = 2 * embedding_dim # Embeddings de time
        self.ban_features_dim = 14 * embedding_dim   # 14 heróis banidos
        self.stats_dim = 50  # 10 jogadores * 5 métricas
        
        self.total_input_dim = (self.radiant_heroes_dim + self.dire_heroes_dim + 
                               self.team_embeddings_dim + self.ban_features_dim + self.stats_dim)
        
        # Encoder
        encoder_layers = []
        input_dim = self.total_input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
            
        encoder_layers.append(nn.Linear(input_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder - reconstrói entrada original diretamente
        decoder_layers = []
        input_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
            
        # Saída final reconstrói entrada original
        decoder_layers.append(nn.Linear(input_dim, self.total_input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Codifica a entrada para o espaço latente."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodifica do espaço latente para as features originais."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass completo."""
        latent = self.encode(x)
        decoded = self.decode(latent)
        return latent, decoded





class DotaDataProcessor:
    """
    Classe para preprocessar dados de partidas de Dota 2 mantendo separação entre times.
    """
    
    def __init__(self, max_heroes: int = 150):
        self.max_heroes = max_heroes
        self.stats_scaler = StandardScaler()
        self.fitted = False
        
    def _process_team_heroes(self, hero_ids: List[int], max_heroes: int = 5) -> np.ndarray:
        """Processa IDs de heróis de um time específico."""
        heroes = np.zeros(max_heroes, dtype=np.int32)
        for i, hero_id in enumerate(hero_ids[:max_heroes]):
            heroes[i] = hero_id if hero_id is not None else 0
        return heroes
    
    def _process_bans(self, ban_ids: List[int], max_bans: int = 14) -> np.ndarray:
        """Processa IDs de heróis banidos."""
        bans = np.zeros(max_bans, dtype=np.int32)
        for i, ban_id in enumerate(ban_ids[:max_bans]):
            bans[i] = ban_id if ban_id is not None else 0
        return bans
    
    def _process_stats(self, radiant_stats: List[List], dire_stats: List[List]) -> np.ndarray:
        """Processa estatísticas dos jogadores mantendo ordem por time."""
        stats = []
        
        # Radiant team stats (5 players * 5 metrics)
        for i in range(5):
            stats.extend([
                radiant_stats[0][i] if i < len(radiant_stats[0]) else 0,  # kills
                radiant_stats[1][i] if i < len(radiant_stats[1]) else 0,  # deaths
                radiant_stats[2][i] if i < len(radiant_stats[2]) else 0,  # assists
                radiant_stats[3][i] if i < len(radiant_stats[3]) else 0,  # gold_per_min
                radiant_stats[4][i] if i < len(radiant_stats[4]) else 0,  # xp_per_min
            ])
        
        # Dire team stats (5 players * 5 metrics)
        for i in range(5):
            stats.extend([
                dire_stats[0][i] if i < len(dire_stats[0]) else 0,  # kills
                dire_stats[1][i] if i < len(dire_stats[1]) else 0,  # deaths
                dire_stats[2][i] if i < len(dire_stats[2]) else 0,  # assists
                dire_stats[3][i] if i < len(dire_stats[3]) else 0,  # gold_per_min
                dire_stats[4][i] if i < len(dire_stats[4]) else 0,  # xp_per_min
            ])
        
        return np.array(stats, dtype=np.float32)
    
    def prepare_data(self, data: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepara dados mantendo separação entre times.
        
        Returns:
            radiant_heroes_tensor: Tensor com IDs dos heróis Radiant
            dire_heroes_tensor: Tensor com IDs dos heróis Dire
            bans_tensor: Tensor com IDs dos heróis banidos
            stats_tensor: Tensor com estatísticas dos jogadores
        """
        radiant_heroes_data = []
        dire_heroes_data = []
        bans_data = []
        stats_data = []
        
        for match in data:
            # Processar heróis SEPARADAMENTE por time
            radiant_heroes = self._process_team_heroes(match['radiant_hero_id'])
            dire_heroes = self._process_team_heroes(match['dire_hero_id'])
            
            radiant_heroes_data.append(radiant_heroes)
            dire_heroes_data.append(dire_heroes)
            
            # Processar bans
            bans = self._process_bans(match['ban_hero_id'])
            bans_data.append(bans)
            
            # Processar estatísticas
            radiant_stats = [
                match['radiant_kills'],
                match['radiant_deaths'],
                match['radiant_assists'],
                match['radiant_gold_per_min'],
                match['radiant_xp_per_min']
            ]
            dire_stats = [
                match['dire_kills'],
                match['dire_deaths'],
                match['dire_assists'],
                match['dire_gold_per_min'],
                match['dire_xp_per_min']
            ]
            
            stats = self._process_stats(radiant_stats, dire_stats)
            stats_data.append(stats)
        
        # Converter para arrays numpy
        radiant_heroes_array = np.array(radiant_heroes_data)
        dire_heroes_array = np.array(dire_heroes_data)
        bans_array = np.array(bans_data)
        stats_array = np.array(stats_data)
        
        # Normalizar estatísticas
        if not self.fitted:
            stats_array = self.stats_scaler.fit_transform(stats_array)
            self.fitted = True
        else:
            stats_array = self.stats_scaler.transform(stats_array)
        
        # Converter para tensors
        radiant_heroes_tensor = torch.LongTensor(radiant_heroes_array)
        dire_heroes_tensor = torch.LongTensor(dire_heroes_array)
        bans_tensor = torch.LongTensor(bans_array)
        stats_tensor = torch.FloatTensor(stats_array)
        
        return radiant_heroes_tensor, dire_heroes_tensor, bans_tensor, stats_tensor
    
    def create_model_input(
        self,
        model: DotaMatchAutoencoder,
        radiant_heroes_tensor: torch.Tensor,
        dire_heroes_tensor: torch.Tensor,
        bans_tensor: torch.Tensor,
        stats_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Cria entrada combinada mantendo separação de times.
        """
        # Detectar o device do modelo
        model_device = next(model.parameters()).device
        
        # Mover todos os tensors para o mesmo device do modelo
        radiant_heroes_tensor = radiant_heroes_tensor.to(model_device)
        dire_heroes_tensor = dire_heroes_tensor.to(model_device)
        bans_tensor = bans_tensor.to(model_device)
        stats_tensor = stats_tensor.to(model_device)
        
        batch_size = radiant_heroes_tensor.size(0)
        
        # Usar torch.no_grad() para evitar problemas de computation graph
        with torch.no_grad():
            # Embeddings dos heróis SEPARADOS por time
            radiant_embedded = model.hero_embedding(radiant_heroes_tensor).view(batch_size, -1)
            dire_embedded = model.hero_embedding(dire_heroes_tensor).view(batch_size, -1)
            
            # Embeddings de time - criar tensor diretamente no device correto
            team_ids = torch.tensor([[0, 1]] * batch_size, device=model_device, dtype=torch.long).view(batch_size, 2)
            team_embedded = model.team_embedding(team_ids).view(batch_size, -1)
            
            # Embeddings dos bans
            bans_embedded = model.hero_embedding(bans_tensor).view(batch_size, -1)
            
            # Concatenar TODAS as features mantendo ordem lógica
            combined_input = torch.cat([
                radiant_embedded,    # Heróis Radiant
                dire_embedded,       # Heróis Dire  
                team_embedded,       # Embeddings de time
                bans_embedded,       # Heróis banidos
                stats_tensor         # Estatísticas
            ], dim=1)
            
            # Retornar tensor detached e clonado para evitar problemas de graph
            return combined_input.clone().detach().requires_grad_(True)


class DotaAutoencoderTrainer:
    """
    Classe para treinar o autoencoder de dados de Dota 2.
    """

    def __init__(
        self,
        model: DotaMatchAutoencoder,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()

    def train_step(self, x: torch.Tensor) -> float:
        """Um passo de treinamento."""
        self.model.train()
        self.optimizer.zero_grad()

        # Garantir que o tensor está no mesmo device do modelo
        device = next(self.model.parameters()).device
        x = x.to(device)

        latent, decoded = self.model(x)

        # Calcular loss de reconstrução simples
        reconstruction_loss = self.criterion(decoded, x)

        reconstruction_loss.backward()
        self.optimizer.step()

        return reconstruction_loss.item()

    def train_epoch(self, dataloader) -> float:
        """Treina uma época completa."""
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            # DataLoader retorna uma tupla, extrair o tensor
            if isinstance(batch, (list, tuple)):
                batch_x = batch[0]
            else:
                batch_x = batch
            loss = self.train_step(batch_x)
            total_loss += loss
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0

    def evaluate(self, dataloader) -> float:
        """Avalia o modelo no conjunto de validação."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # DataLoader retorna uma tupla, extrair o tensor
                if isinstance(batch, (list, tuple)):
                    batch_x = batch[0]
                else:
                    batch_x = batch
                
                # Garantir que o tensor está no mesmo device do modelo
                device = next(self.model.parameters()).device
                batch_x = batch_x.to(device)
                
                latent, decoded = self.model(batch_x)
                loss = self.criterion(decoded, batch_x)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0


def calculate_max_heroes(dataset_path: str) -> int:
    """
    Calcula o número máximo de heróis no dataset.
    
    Args:
        dataset_path: Caminho para o dataset baixado
        
    Returns:
        Número máximo de heróis disponíveis no dataset
    """
    max_hero_id = pl.read_csv(
        f"{dataset_path}/{heroes_file}"
    ).select("id").max().item()
    
    return int(max_hero_id)


def create_autoencoder_from_dataset(dataset_path: str, 
                                   embedding_dim: int = 32,
                                   hidden_dims: List[int] = [256, 128, 64],
                                   latent_dim: int = 32,
                                   dropout_rate: float = 0.2):
    """
    Cria um autoencoder e processador com o número correto de heróis do dataset.
    
    Args:
        dataset_path: Caminho para o dataset
        embedding_dim: Dimensão dos embeddings
        hidden_dims: Dimensões das camadas ocultas
        latent_dim: Dimensão do espaço latente
        dropout_rate: Taxa de dropout
        
    Returns:
        Tupla (modelo, processador) configurados para o dataset
    """
    max_heroes = calculate_max_heroes(dataset_path)
    
    model = DotaMatchAutoencoder(
        max_heroes=max_heroes,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate
    )
    processor = DotaDataProcessor(max_heroes=max_heroes)
    
    return model, processor


if __name__ == "__main__":

    # Download do dataset
    dataset_name = "bwandowando/dota-2-pro-league-matches-2023/versions/177"
    path = kagglehub.dataset_download(dataset_name)

    # Calcular o número máximo de heróis do dataset
    max_heroes = calculate_max_heroes(path)
    print(f"Número máximo de heróis no dataset: {max_heroes}")

    dataset = get_dataset(path, patches=[54], tier=["professional"],
                          min_duration=10 * 60, max_duration=120 * 60)

    # Criar autoencoder com separação de times
    print("\n=== Testando DotaMatchAutoencoder (team-separated) ===")
    model = DotaMatchAutoencoder(max_heroes=max_heroes)
    processor = DotaDataProcessor(max_heroes=max_heroes)
    print(f"Modelo - Max Heroes: {model.max_heroes}")
    
    # Testar processamento de dados
    data = processor.prepare_data(dataset.to_dicts())
    model_input = processor.create_model_input(model, *data)
    
    print(f"Forma da entrada do modelo: {model_input.shape}")
    print(f"Dimensões esperadas: {model.total_input_dim}")
    
    # Testar forward pass
    latent, decoded = model(model_input)
    print(f"Forma do espaço latente: {latent.shape}")
    print(f"Chaves decodificadas: {decoded.keys()}")
    
    # Demonstrar como usar a função de conveniência
    print("\n=== Testando função de conveniência ===")
    model_convenience, processor_convenience = create_autoencoder_from_dataset(path)
    print(f"Modelo criado via função - Max Heroes: {model_convenience.max_heroes}")
