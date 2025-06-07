import sys
import kagglehub
from dataset import get_dataset, save_dataset
from heroes import get_heroes
from model import Dota2Autoencoder
from plot import plot_loss_history
from pathlib import Path


def main():
    # Carregar metadados dos heróis
    dataset_name = "bwandowando/dota-2-pro-league-matches-2023"
    path = kagglehub.dataset_download(dataset_name)
    print("="*50)
    dataset, player_cols, hero_cols = get_dataset(
        path, specific_patches=[54])  # 54 é o patch 6.7

    if ("--save" in sys.argv):
        save_dataset(dataset.head())
        return

    if ("--save-all" in sys.argv):
        save_dataset(dataset, output_path="./tmp/DATASET_FULL.json")
        return

    heroes, _, _, dict_roles = get_heroes(path)
    n_heroes = heroes.select("hero_id").max().collect().item()
    n_hero_stats = len(dict_roles) + len(hero_cols)
    n_player_stats = len(player_cols)

    scale = 0.5
    train_data = dataset.sample(fraction=0.75 * scale, seed=42, shuffle=True)
    validation_data = dataset.sample(
        fraction=0.15 * scale, seed=42, shuffle=True)
    test_data = dataset.sample(fraction=0.15 * scale, seed=42, shuffle=True)
    save_model_path = "./tmp/dota2_autoencoder.h5"
    save_loss_history_path = "./tmp/dota2_autoencoder_loss_history.csv"

    print("="*50)
    print(
        f"Conjunto de dados carregado com {dataset.shape[0]} linhas e {dataset.shape[1]} colunas.")
    print(
        f"Tamanho dos dados de treino: {train_data.shape[0]}, Tamanho dos dados de validação: {validation_data.shape[0]}, Tamanho dos dados de teste: {test_data.shape[0]}")
    print(
        f"Total de heróis: {n_heroes}, Total de estatísticas de heróis: {n_hero_stats}, Total de estatísticas de jogadores: {n_player_stats}")

    autoencoder = Dota2Autoencoder(
        hero_pick_embedding_dim=16,
        hero_role_embedding_dim=8,
        n_player_stats=n_player_stats,
        n_heroes=n_heroes + 1,
        dict_roles=dict_roles,
        n_players=5,
        n_bans=7,
        latent_dim=32,
        hidden_layers=[256, 128, 64],
        dropout=0.3,
        learning_rate=0.001,
        verbose=True
    )

    print("="*50)
    print("Treinando Dota2 Autoencoder...")
    autoencoder.train_data(
        training_df=train_data, validation_df=validation_data, epochs=20, verbose=True)
    autoencoder.save_model(save_model_path)
    autoencoder.save_loss_history(save_loss_history_path)
    print(
        f"Modelo salvo em {save_model_path} e histórico de perda salvo em {save_loss_history_path}.")
    print("Treinamento concluído.")

    csv_path = "./tmp/dota2_autoencoder_loss_history.csv"
    save_path = "./tmp/dota2_autoencoder_loss_plot.png"

    accuracy = autoencoder.test_model(test_data)
    print(f"Accuracy: {accuracy:.4f}")
    print("Teste do modelo concluído.")

    # Plotar histórico de perda
    print(f"Plotando histórico de perda em {save_path}...")
    plot_loss_history(csv_path, save_path,
                      title="Dota 2 Autoencoder Loss History")


if __name__ == "__main__":
    main()
