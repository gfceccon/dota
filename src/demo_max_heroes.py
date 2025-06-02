"""
Demonstração de como usar o parâmetro max_heroes no autoencoder de Dota 2.

Este script mostra como:
1. Calcular o número máximo de heróis do dataset
2. Criar autoencoder com o parâmetro correto
3. Usar as funções de conveniência para configuração automática
"""

import kagglehub
import polars as pl
import json
from autoencoders import (
    DotaMatchAutoencoder,
    DotaDataProcessor,
    calculate_max_heroes,
    create_autoencoder_from_dataset
)
from files import heroes_file
from dataset import get_dataset


def demo_max_heroes_calculation():
    """Demonstra como calcular max_heroes do dataset."""
    print("=== Calculando max_heroes do dataset ===")
    
    # Download do dataset
    dataset_name = "bwandowando/dota-2-pro-league-matches-2023/versions/177"
    path = kagglehub.dataset_download(dataset_name)
    
    # Método 1: Usando a função utilitária
    max_heroes = calculate_max_heroes(path)
    print(f"Max heroes (função utilitária): {max_heroes}")
    
    # Método 2: Cálculo manual
    max_heroes_manual = pl.read_csv(
        f"{path}/{heroes_file}"
    ).select("id").max().item()
    print(f"Max heroes (manual): {max_heroes_manual}")
    
    return path, max_heroes


def demo_manual_creation(path: str, max_heroes: int):
    """Demonstra criação manual do modelo com max_heroes."""
    print(f"\n=== Criação manual com max_heroes={max_heroes} ===")
    
    # Criar modelo e processador manualmente
    model = DotaMatchAutoencoder(max_heroes=max_heroes)
    processor = DotaDataProcessor(max_heroes=max_heroes)
    
    print(f"Modelo configurado para {model.max_heroes} heróis")
    print(f"Processador configurado para {processor.max_heroes} heróis")
    
    # Mostrar dimensões do modelo
    print(f"Dimensões de entrada: {model.total_input_dim}")
    
    return model, processor


def demo_convenience_function(path: str):
    """Demonstra uso da função de conveniência."""
    print(f"\n=== Usando função de conveniência ===")
    
    # Criar modelo usando a função de conveniência
    model, processor = create_autoencoder_from_dataset(path)
    
    print(f"Modelo (conveniência): {model.max_heroes} heróis")
    
    # Criar com parâmetros customizados
    model_custom, processor_custom = create_autoencoder_from_dataset(
        path, 
        embedding_dim=64,
        hidden_dims=[512, 256, 128],
        latent_dim=64,
        dropout_rate=0.3
    )
    
    print(f"Modelo customizado: {model_custom.max_heroes} heróis, "
          f"embedding_dim={model_custom.embedding_dim}, "
          f"latent_dim={model_custom.latent_dim}")
    
    return model, processor


def demo_data_processing(path: str, model: DotaMatchAutoencoder, processor: DotaDataProcessor):
    """Demonstra processamento de dados com max_heroes correto."""
    print(f"\n=== Processamento de dados ===")
    
    # Carregar dados de exemplo
    with open('/home/seduq/Github/dota/processed_dataset.json', 'r') as f:
        sample_data = json.load(f)
    
    print(f"Processando {len(sample_data)} partidas de exemplo...")
    
    # Processar dados
    radiant_heroes, dire_heroes, bans, stats = processor.prepare_data(sample_data)
    
    print(f"Forma heróis Radiant: {radiant_heroes.shape}")
    print(f"Forma heróis Dire: {dire_heroes.shape}")
    print(f"Forma bans: {bans.shape}")
    print(f"Forma stats: {stats.shape}")
    
    # Criar entrada do modelo
    model_input = processor.create_model_input(model, radiant_heroes, dire_heroes, bans, stats)
    print(f"Forma entrada do modelo: {model_input.shape}")
    
    # Forward pass
    latent, decoded = model(model_input)
    print(f"Forma espaço latente: {latent.shape}")
    print(f"Chaves decodificadas: {list(decoded.keys())}")
    
    for key, tensor in decoded.items():
        print(f"  {key}: {tensor.shape}")


def demo_comparison_old_vs_new():
    """Demonstra diferença entre uso com e sem max_heroes parametrizado."""
    print(f"\n=== Comparação: antes vs depois ===")
    
    print("ANTES (hard-coded):")
    print("  model = DotaMatchAutoencoder(num_heroes=150)  # Fixo!")
    print("  processor = DotaDataProcessor()  # Sem parâmetro!")
    
    print("\nDEPOIS (parametrizado):")
    print("  max_heroes = calculate_max_heroes(dataset_path)")
    print("  model = DotaMatchAutoencoder(max_heroes=max_heroes)")
    print("  processor = DotaDataProcessor(max_heroes=max_heroes)")
    
    print("\nOU usando função de conveniência:")
    print("  model, processor = create_autoencoder_from_dataset(dataset_path)")


if __name__ == "__main__":
    # Executar todas as demonstrações
    path, max_heroes = demo_max_heroes_calculation()
    
    model, processor = demo_manual_creation(path, max_heroes)
    
    model_conv, processor_conv = demo_convenience_function(path)
    
    demo_data_processing(path, model, processor)
    
    demo_comparison_old_vs_new()
    
    print(f"\n=== Resumo ===")
    print(f"✅ Parâmetro max_heroes adicionado a todas as classes")
    print(f"✅ Função calculate_max_heroes() para calcular do dataset")
    print(f"✅ Função create_autoencoder_from_dataset() para conveniência")
    print(f"✅ Compatibilidade com dataset real (max_heroes={max_heroes})")
    print(f"✅ Remoção completa da versão V1 - apenas team-separated version mantida")