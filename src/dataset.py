import kagglehub
import polars as pl
from patches import get_patches
from preprocessing import preprocess_dota_dataset


def get_dataset(path: str, patches: list[int], tier: list[str] = ["professional"], min_duration: int = 10 * 60) -> pl.DataFrame:
    """
    Função principal para carregar e pré-processar o dataset de Dota 2.
    """
    
    dataset = preprocess_dota_dataset(
        path=path,
        patches=patches,
        tier=tier,
        min_duration=min_duration
    )
    
    return dataset

def save_dataset(dataset: pl.DataFrame, output_path: str):
    """
    Salva o dataset pré-processado.

    Args:
        dataset: Dataset pré-processado
        output_path: Caminho para salvar o arquivo
    """
    print(f"Salvando dataset em {output_path}...")
    dataset.write_json(output_path)
    print("Dataset salvo com sucesso!")


if __name__ == "__main__":

    # Download do dataset
    dataset_name = "bwandowando/dota-2-pro-league-matches-2023/versions/177"
    path = kagglehub.dataset_download(dataset_name)

    # Definir patches e tier
    patches = get_patches(path, begin_year=2023, end_year=2024) # Patches disponíveis
    tier = ["professional"] # Tier de interesse
    min_duration=10 * 60  # 10 minutos em segundos

    print(f"Patches disponíveis: {patches}")
    sum = 0
    for patch, (count, _) in patches.items():
        sum += count
    print(f"- Total de jogos: {sum}")

    
    # Carregar e pré-processar o dataset
    dataset = get_dataset(path, list(patches.keys()), tier, min_duration)
    
    # Mostrar informações do dataset
    print(f"\nDataset pré-processado:")
    print(f"- Número de matches: {len(dataset)}")
    print(f"- Número de features: {len(dataset.columns)}")
    print(f"- Colunas: {dataset.columns[:10]}...")  # Primeiras 10 colunas

    sample = dataset.sample()
    print(f"\nExibindo dados do match_id {sample["match_id"][0]}:")
    print(sample)
    
    # Salvar dataset processado
    output_path = "/home/seduq/Github/dota/processed_dataset.json"
    save_dataset(dataset, output_path)