import kagglehub
from dataset import get_dataset, save_dataset
from heroes import get_heroes
from model import Dota2Autoencoder
import sys


def main():
    # Carregar metadados dos heróis
    dataset_name = "bwandowando/dota-2-pro-league-matches-2023"
    path = kagglehub.dataset_download(dataset_name)
    print("="*50)
    dataset, player_cols, hero_cols = get_dataset(path, specific_patches=[54]) # 54 é o patch 6.7
    print("="*50)
    
    if("--save" in sys.argv):
        save_dataset(dataset.head())
        return
    
    if("--save-all" in sys.argv):
        save_dataset(dataset, output_path="./tmp/DATASET_FULL.json")
        return
    
    heroes, _, _, dict_roles = get_heroes(path)
    n_heroes = heroes.select("hero_id").max().collect().item()
    n_hero_stats = len(dict_roles) + len(hero_cols) 
    n_player_stats = len(player_cols)
    
    train_data = dataset.sample(fraction=0.9, seed=42, shuffle=True)
    validation_data = dataset.sample(fraction=0.1, seed=42, shuffle=True)
    save_model_path = "./tmp/dota2_autoencoder.h5"
    save_loss_history_path = "./tmp/dota2_autoencoder_loss_history.csv"
    
    print("="*50)
    print(f"Conjunto de dados carregado com {dataset.shape[0]} linhas e {dataset.shape[1]} colunas.")
    print(f"Tamanho dos dados de treino: {train_data.shape}, Tamanho dos dados de validação: {validation_data.shape}")
    print(f"Total de estatísticas de heróis: {n_hero_stats}, Total de estatísticas de jogadores: {n_player_stats}")
    print(f"Total de heróis: {n_heroes}, Total de estatísticas de herói: {n_hero_stats}")
    print("="*50)

    
    
    
    autoencoder = Dota2Autoencoder(
        hero_pick_embedding_dim=16,
        hero_role_embedding_dim=8,
        n_player_stats=n_player_stats,
        n_heroes=n_heroes + 1,
        dict_roles=dict_roles,
        n_players=5,
        n_bans=7,
        latent_dim=16,
        hidden_layers=[128, 32],
        dropout=0.2,
        learning_rate=0.001,
        verbose=True
    )
    
    print("="*50)
    print("Treinando Dota2 Autoencoder...")
    autoencoder.train_data(training_df=train_data, validation_df=validation_data, epochs=50, verbose=True)
    autoencoder.save_model(save_model_path)
    autoencoder.save_loss_history(save_loss_history_path)
    print(f"Modelo salvo em {save_model_path} e histórico de perda salvo em {save_loss_history_path}.")
    print("Treinamento concluído.")
    print("="*50)
if __name__ == "__main__":
    main()
