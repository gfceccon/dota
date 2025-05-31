import kagglehub
import polars as pl
from preprocessing import preprocess_dota_dataset, save_preprocessed_dataset


def main():
    """
    Função principal para carregar e pré-processar o dataset de Dota 2.
    """
    # Download do dataset
    dataset_name = "bwandowando/dota-2-pro-league-matches-2023/versions/177"
    path = kagglehub.dataset_download(dataset_name)
    
    print(f"Dataset baixado em: {path}")
    
    # Pré-processar dados
    print("Iniciando pré-processamento...")
    dataset = preprocess_dota_dataset(
        path=path,
        patches=[54],  # Patch 7.34
        tier=["professional"],
        min_duration=10 * 60  # 10 minutos mínimos
    )
    
    # Mostrar informações do dataset
    dataset_collected = dataset.collect()
    print(f"\nDataset pré-processado:")
    print(f"- Número de matches: {len(dataset_collected)}")
    print(f"- Número de features: {len(dataset_collected.columns)}")
    print(f"- Colunas: {dataset_collected.columns[:10]}...")  # Primeiras 10 colunas
    
    # Salvar dataset processado
    output_path = "/home/seduq/Github/dota/processed_dataset.csv"
    save_preprocessed_dataset(dataset, output_path)
    
    return dataset_collected


if __name__ == "__main__":
    dataset = main()

