import ast
import os
import sys
import kagglehub
import torch
from heroes import get_heroes
from dataset import get_dataset, save_dataset
from patches import get_patches
from model import Dota2Autoencoder
import polars as pl


def train_autoencoder(train_data, validation_data, test_data, dict_roles,
                      hero_cols, player_cols, match_cols,
                      save_filename, save_model_path, save_loss_history_path,
                      n_heroes, epochs=100,):

    n_hero_stats = len(dict_roles) + len(hero_cols)
    n_player_stats = len(player_cols)

    print("="*50)
    print(
        f"Tamanho dos dados de treino: {train_data.shape[0]}, Tamanho dos dados de validação: {validation_data.shape[0]}, Tamanho dos dados de teste: {test_data.shape[0]}")
    print(
        f"Total de estatísticas de heróis: {n_hero_stats}, Total de estatísticas de jogadores: {n_player_stats}")

    autoencoder = Dota2Autoencoder(
        hero_pick_embedding_dim=16,
        hero_role_embedding_dim=8,
        dict_roles=dict_roles,
        hero_cols=hero_cols,
        player_cols=player_cols,
        match_cols=match_cols,
        n_heroes=n_heroes + 1,
        n_players=5,
        n_bans=7,
        latent_dim=4,
        hidden_layers=[256, 64],
        dropout=0.3,
        learning_rate=0.001,
        verbose=False
    )

    print("="*50)
    print("Treinando Dota2 Autoencoder...")
    autoencoder.train_data(
        training_df=train_data, validation_df=validation_data, epochs=epochs, best_model_filename=save_filename, verbose=False)
    autoencoder.save_loss_history(save_loss_history_path)
    print(
        f"Modelo salvo em {save_model_path} e histórico de perda salvo em {save_loss_history_path}.")
    print("Treinamento concluído.")
    return autoencoder, train_data, validation_data, test_data


def load_or_prepare_dataset(dataset_full_path: str, dataset_metadata_path: str, patches: list[int] = []):
    if (os.path.exists(dataset_full_path) and
            os.path.exists(dataset_metadata_path)):
        print("Carregando modelo e metadados do dataset...")
        dataset = pl.read_json(dataset_full_path)
        dataset_metadata = pl.read_json(dataset_metadata_path)
        n_heroes = dataset_metadata["n_heroes"].item()
        n_hero_stats = dataset_metadata["n_hero_stats"].item()
        n_player_stats = dataset_metadata["n_player_stats"].item()
        player_cols = dataset_metadata["player_cols"].item()
        match_cols = dataset_metadata["match_cols"].item()
        hero_cols = dataset_metadata["hero_cols"].item()
        dict_roles = dataset_metadata["dict_roles"].item()
        patches_info = ast.literal_eval(
            dataset_metadata["patches_info"].item())
        print("Dataset já carregado do arquivo.")
        return dataset, player_cols, match_cols, hero_cols, dict_roles, n_heroes, n_hero_stats, n_player_stats, patches_info, False
    else:
        dataset_name = "bwandowando/dota-2-pro-league-matches-2023"
        path = kagglehub.dataset_download(dataset_name)
        dataset, player_cols, match_cols, hero_cols = get_dataset(
            path, specific_patches=patches)
        heroes, _, _, dict_roles = get_heroes(path)
        n_heroes = heroes.select("hero_idx").count().collect().item()
        n_hero_stats = len(dict_roles) + len(hero_cols)
        n_player_stats = len(player_cols)
        patches_info = get_patches(path)
        return dataset, player_cols, match_cols, hero_cols, dict_roles, n_heroes, n_hero_stats, n_player_stats, patches_info, True


def save_dataset_and_metadata(
        dataset, player_cols, match_cols, hero_cols, dict_roles, n_heroes, n_hero_stats, n_player_stats, patches_info,
        dataset_full_path: str, dataset_metadata_path: str,):
    save_dataset(dataset, output_path=dataset_full_path)
    dataset_metadata = pl.DataFrame({
        "n_heroes": [n_heroes],
        "n_hero_stats": [n_hero_stats],
        "n_player_stats": [n_player_stats],
        "player_cols": [player_cols],
        "match_cols": [match_cols],
        "hero_cols": [hero_cols],
        "dict_roles": [dict_roles],
        "patches_info": [str(patches_info)]
    })
    dataset_metadata.write_json(dataset_metadata_path)
    print(f"Dataset completo salvo em {dataset_full_path}.")
    print(f"Metadados do dataset salvos em {dataset_metadata_path}.")


def prepare_data_splits(dataset, df_scale=1.0):
    train_data = dataset.sample(
        fraction=0.75 * df_scale, seed=42, shuffle=True)
    validation_data = dataset.sample(
        fraction=0.15 * df_scale, seed=42, shuffle=True)
    test_data = dataset.sample(
        fraction=0.15 * df_scale, seed=42, shuffle=True)
    return train_data, validation_data, test_data


def train_or_load_autoencoder(dataset, dict_roles, hero_cols, player_cols, match_cols,
                              n_heroes, save_model_path, save_loss_history_path, train=False):
    if (os.path.exists(save_model_path) and train is False):
        print("Carregando modelo treinado...")
        autoencoder = Dota2Autoencoder.load_model(
            "./tmp/dota2_autoencoder.h5",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return autoencoder, None, None, None
    else:
        print("Treinando novo modelo de autoencoder...")
        save_filenames = get_filenames([1])
        train_data, validation_data, test_data = prepare_data_splits(
            dataset, 1.0)
        autoencoder, train_data, validation_data, test_data = train_autoencoder(
            train_data=train_data,
            validation_data=validation_data,
            test_data=test_data,
            dict_roles=dict_roles,
            hero_cols=hero_cols,
            player_cols=player_cols,
            match_cols=match_cols,
            n_heroes=n_heroes,
            save_filename=save_filenames["save_filename"],
            save_model_path=save_model_path,
            save_loss_history_path=save_loss_history_path,
        )
        return autoencoder, train_data, validation_data, test_data


def get_filenames(patches: list[int]) -> dict[str, str]:
    patches_str = [str(patch) for patch in patches]
    save_filename = f"dota2_autoencoder_{"_".join(patches_str)}"
    return {
        "dataset_full_path": f"./tmp/{save_filename}_dataset.json",
        "dataset_metadata_path": f"./tmp/{save_filename}_metadata.json",
        "save_model_path": f"./tmp/{save_filename}_model.h5",
        "save_loss_history_path": f"./tmp/{save_filename}_history.csv",
        "save_plot_loss_history_path": f"./tmp/{save_filename}_history.png",
        "report_path": "./tmp/report.txt",
        "save_filename": save_filename
    }
