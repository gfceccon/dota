#!/usr/bin/env python3
"""
Exemplo de uso do DotaMatchAutoencoder

Este arquivo demonstra como usar o autoencoder para:
1. Carregar e preprocessar dados de partidas de Dota 2
2. Treinar o modelo
3. Extrair representações latentes
4. Analisar similaridade entre partidas
5. Detectar anomalias
6. Reconstruir dados de partidas

Autor: Exemplo de uso
Data: 2025-06-02
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import kagglehub

# Importar as classes do autoencoder
from autoencoders import (
    DotaMatchAutoencoder,
    DotaDataProcessor,
    DotaAutoencoderTrainer,
    create_autoencoder_from_dataset,
    calculate_max_heroes
)
from src.dataset import get_dataset


def setup_device():
    """Configura o dispositivo para treinamento (GPU se disponível)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    return device


def load_and_prepare_data(dataset_path: str, sample_size: int = 1000):
    """
    Carrega e prepara os dados do dataset.
    
    Args:
        dataset_path: Caminho para o dataset
        sample_size: Número de amostras para usar (None para usar todas)
    
    Returns:
        Dados preprocessados prontos para treinamento
    """
    print("🔄 Carregando dataset...")
    
    # Carregar dataset completo
    dataset = get_dataset(
        dataset_path, 
        patches=[54], 
        tier=["professional"],
        min_duration=10 * 60, 
        max_duration=120 * 60
    )
    
    print(f"📊 Dataset carregado com {len(dataset)} partidas")
    
    # Usar uma amostra se especificado
    if sample_size and len(dataset) > sample_size:
        dataset = dataset.sample(sample_size, seed=42)
        print(f"📊 Usando amostra de {len(dataset)} partidas")
    
    return dataset.to_dicts()


def create_data_loaders(model_input: torch.Tensor, batch_size: int = 32, train_split: float = 0.8, device=None):
    """
    Cria data loaders para treinamento e validação.
    
    Args:
        model_input: Tensor com dados preprocessados
        batch_size: Tamanho do batch
        train_split: Proporção dos dados para treinamento
        device: Dispositivo para colocar os dados
    
    Returns:
        train_loader, val_loader
    """
    # Mover dados para o device se especificado
    if device:
        model_input = model_input.to(device)
    
    dataset_size = len(model_input)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    # Dividir dados
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_data = model_input[train_indices]
    val_data = model_input[val_indices]
    
    # Criar datasets e dataloaders
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📦 Dados divididos: {train_size} treino, {val_size} validação")
    
    return train_loader, val_loader


def train_autoencoder(model, trainer, train_loader, val_loader, epochs: int = 50, device=None):
    """
    Treina o autoencoder.
    
    Args:
        model: Modelo do autoencoder
        trainer: Objeto trainer
        train_loader: DataLoader de treinamento
        val_loader: DataLoader de validação
        epochs: Número de épocas
        device: Dispositivo para treinamento
    
    Returns:
        Histórico de treinamento
    """
    if device:
        model = model.to(device)
    
    train_losses = []
    val_losses = []
    
    print(f"🚀 Iniciando treinamento por {epochs} épocas...")
    
    for epoch in range(epochs):
        # Treinar
        train_loss = trainer.train_epoch(train_loader)
        
        # Validar
        val_loss = trainer.evaluate(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Época {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    print("✅ Treinamento concluído!")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses
    }


def plot_training_history(history, save_path: str):
    """
    Plota o histórico de treinamento.
    
    Args:
        history: Dicionário com histórico de perdas
        save_path: Caminho para salvar o gráfico
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    plt.plot(epochs, history['train_losses'], 'b-', label='Treino', linewidth=2)
    plt.plot(epochs, history['val_losses'], 'r-', label='Validação', linewidth=2)
    
    plt.title('Histórico de Treinamento do Autoencoder', fontsize=16)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico salvo em: {save_path}")
    
    plt.show()


def extract_latent_representations(model, data_loader, device=None):
    """
    Extrai representações latentes de todos os dados.
    
    Args:
        model: Modelo treinado
        data_loader: DataLoader com dados
        device: Dispositivo de computação
    
    Returns:
        Array numpy com representações latentes
    """
    model.eval()
    latent_representations = []
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                batch_x = batch[0]
            else:
                batch_x = batch
                
            # Garantir que está no device correto
            if device:
                batch_x = batch_x.to(device)
            
            latent = model.encode(batch_x)
            latent_representations.append(latent.cpu().numpy())
    
    return np.vstack(latent_representations)


def analyze_latent_space(latent_representations, save_path: str):
    """
    Analisa o espaço latente usando PCA.
    
    Args:
        latent_representations: Representações latentes
        save_path: Caminho para salvar o gráfico
    """
    print("🔍 Analisando espaço latente...")
    
    # Aplicar PCA para visualização 2D
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_representations)
    
    # Plotar
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=20)
    plt.title('Visualização do Espaço Latente (PCA)', fontsize=16)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} da variância)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} da variância)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualização salva em: {save_path}")
    
    plt.show()
    
    print(f"📈 Variância explicada pelos 2 primeiros componentes: "
          f"{sum(pca.explained_variance_ratio_):.2%}")


def find_similar_matches(latent_representations, match_index: int, top_k: int = 5):
    """
    Encontra partidas similares baseadas na representação latente.
    
    Args:
        latent_representations: Representações latentes
        match_index: Índice da partida de referência
        top_k: Número de partidas similares para retornar
    
    Returns:
        Índices das partidas mais similares e suas similaridades
    """
    # Calcular similaridade coseno
    target_match = latent_representations[match_index:match_index+1]
    similarities = cosine_similarity(target_match, latent_representations)[0]
    
    # Encontrar mais similares (excluindo a própria partida)
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
    similar_scores = similarities[similar_indices]
    
    print(f"🎯 Partidas mais similares à partida {match_index}:")
    for i, (idx, score) in enumerate(zip(similar_indices, similar_scores)):
        print(f"  {i+1}. Partida {idx} - Similaridade: {score:.3f}")
    
    return similar_indices, similar_scores


def detect_anomalies(latent_representations, threshold_percentile: float = 95):
    """
    Detecta anomalias baseadas na distância no espaço latente.
    
    Args:
        latent_representations: Representações latentes
        threshold_percentile: Percentil para definir anomalias
    
    Returns:
        Índices das partidas anômalas
    """
    # Calcular distâncias do centroide
    centroid = np.mean(latent_representations, axis=0)
    distances = np.linalg.norm(latent_representations - centroid, axis=1)
    
    # Definir threshold
    threshold = np.percentile(distances, threshold_percentile)
    anomaly_indices = np.where(distances > threshold)[0]
    
    print(f"🚨 Detectadas {len(anomaly_indices)} partidas anômalas "
          f"(>{threshold_percentile}º percentil)")
    
    if len(anomaly_indices) > 0:
        print("Partidas anômalas:")
        for idx in anomaly_indices[:10]:  # Mostrar apenas as primeiras 10
            print(f"  - Partida {idx} - Distância: {distances[idx]:.3f}")
    
    return anomaly_indices


def reconstruction_analysis(model, sample_data, device=None):
    """
    Analisa a qualidade da reconstrução.
    
    Args:
        model: Modelo treinado
        sample_data: Amostra de dados para testar
        device: Dispositivo de computação
    """
    model.eval()
    
    with torch.no_grad():
        # Garantir que os dados estão no device correto
        if device:
            sample_data = sample_data.to(device)
        
        latent, reconstructed = model(sample_data)
        
        # Calcular erro de reconstrução
        mse = nn.MSELoss()(reconstructed, sample_data)
        mae = nn.L1Loss()(reconstructed, sample_data)
        
        print(f"📊 Qualidade da Reconstrução:")
        print(f"  - MSE: {mse.item():.4f}")
        print(f"  - MAE: {mae.item():.4f}")
        
        # Análise por componente (opcional)
        diff = torch.abs(reconstructed - sample_data)
        print(f"  - Erro médio por feature: {diff.mean(dim=0).mean().item():.4f}")
        print(f"  - Erro máximo: {diff.max().item():.4f}")


def save_model(model, processor, filepath: str):
    """
    Salva o modelo e processador treinados.
    
    Args:
        model: Modelo treinado
        processor: Processador de dados
        filepath: Caminho para salvar
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'max_heroes': model.max_heroes,
            'embedding_dim': model.embedding_dim,
            'latent_dim': model.latent_dim,
        },
        'processor_scaler': processor.stats_scaler,
        'processor_fitted': processor.fitted
    }, filepath)
    
    print(f"💾 Modelo salvo em: {filepath}")


def load_model(filepath: str, device=None):
    """
    Carrega um modelo salvo.
    
    Args:
        filepath: Caminho do modelo salvo
        device: Dispositivo para carregar
    
    Returns:
        Modelo e processador carregados
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # Recriar modelo
    config = checkpoint['model_config']
    model = DotaMatchAutoencoder(
        max_heroes=config['max_heroes'],
        embedding_dim=config['embedding_dim'],
        latent_dim=config['latent_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Recriar processador
    processor = DotaDataProcessor(max_heroes=config['max_heroes'])
    processor.stats_scaler = checkpoint['processor_scaler']
    processor.fitted = checkpoint['processor_fitted']
    
    if device:
        model = model.to(device)
    
    print(f"📂 Modelo carregado de: {filepath}")
    
    return model, processor


def main():
    """
    Função principal que demonstra o uso completo do autoencoder.
    """
    print("🎮 Exemplo de Uso do Dota Match Autoencoder")
    print("=" * 50)
    
    # 1. Configuração inicial
    device = setup_device()
    
    # 2. Download e carregamento do dataset
    print("\n📦 1. Baixando dataset...")
    dataset_name = "bwandowando/dota-2-pro-league-matches-2023/versions/177"
    dataset_path = kagglehub.dataset_download(dataset_name)
    
    # 3. Carregamento dos dados
    print("\n📊 2. Carregando e preparando dados...")
    raw_data = load_and_prepare_data(dataset_path, sample_size=500)  # Amostra pequena para exemplo
    
    # 4. Criação do modelo e processador
    print("\n🧠 3. Criando modelo e processador...")
    model, processor = create_autoencoder_from_dataset(dataset_path)
    model = model.to(device)  # Mover modelo para o device
    print(f"Modelo criado com {model.max_heroes} heróis máximos")
    
    # 5. Preprocessamento dos dados
    print("\n⚙️ 4. Preprocessando dados...")
    tensors = processor.prepare_data(raw_data)
    model_input = processor.create_model_input(model, *tensors)
    print(f"Entrada do modelo: {model_input.shape}")
    
    # 6. Criação dos data loaders
    print("\n📦 5. Criando data loaders...")
    train_loader, val_loader = create_data_loaders(model_input, batch_size=16, device=device)
    
    # 7. Treinamento
    print("\n🚀 6. Treinando modelo...")
    trainer = DotaAutoencoderTrainer(model, learning_rate=1e-3)
    history = train_autoencoder(model, trainer, train_loader, val_loader, epochs=20, device=device)
    
    # 8. Visualização do treinamento
    print("\n📊 7. Visualizando resultados do treinamento...")
    plot_training_history(history, "training_history.png")
    
    # 9. Extração de representações latentes
    print("\n🔍 8. Extraindo representações latentes...")
    # Combinar train e val loaders para análise completa
    full_loader = DataLoader(TensorDataset(model_input.to(device)), batch_size=32, shuffle=False)
    latent_representations = extract_latent_representations(model, full_loader, device)
    print(f"Representações latentes extraídas: {latent_representations.shape}")
    
    # 10. Análise do espaço latente
    print("\n🎯 9. Analisando espaço latente...")
    analyze_latent_space(latent_representations, "latent_space_pca.png")
    
    # 11. Busca por similaridade
    print("\n🔎 10. Buscando partidas similares...")
    similar_indices, similarities = find_similar_matches(latent_representations, match_index=0, top_k=5)
    
    # 12. Detecção de anomalias
    print("\n🚨 11. Detectando anomalias...")
    anomaly_indices = detect_anomalies(latent_representations, threshold_percentile=95)
    
    # 13. Análise de reconstrução
    print("\n📊 12. Analisando qualidade da reconstrução...")
    sample_data = model_input[:10].to(device)  # Usar primeiras 10 amostras e mover para device
    reconstruction_analysis(model, sample_data, device)
    
    # 14. Salvamento do modelo
    print("\n💾 13. Salvando modelo...")
    save_model(model, processor, "dota_autoencoder_model.pth")
    
    # 15. Demonstração de carregamento
    print("\n📂 14. Demonstrando carregamento do modelo...")
    loaded_model, loaded_processor = load_model("dota_autoencoder_model.pth", device)
    
    print("\n✅ Exemplo completo executado com sucesso!")
    print("\nArquivos gerados:")
    print("  - training_history.png: Histórico de treinamento")
    print("  - latent_space_pca.png: Visualização do espaço latente")
    print("  - dota_autoencoder_model.pth: Modelo treinado")


if __name__ == "__main__":
    # Configurar para reprodutibilidade
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Execução interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        raise
