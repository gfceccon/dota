import ast
import os
import sys
import kagglehub
import torch
from heroes import get_heroes
from dataset import get_dataset, save_dataset
from patches import get_patches
from plot import plot_loss_history
from model import Dota2Autoencoder
import polars as pl

patches = [53]
patches_str = [str(patch) for patch in patches]
dataset_path = f"./tmp/DATASET_{"_".join(patches_str)}.json"
dataset_full_path = f"./tmp/DATASET_FULL_{"_".join(patches_str)}.json"
dataset_metadata_path = f"./tmp/DATASET_METADATA_{"_".join(patches_str)}.json"
save_filename = f"dota2_autoencoder_{"_".join(patches_str)}"
save_model_path = f"./tmp/{save_filename}_model.h5"
save_loss_history_path = f"./tmp/{save_filename}_history.csv"
save_plot_loss_history_path = f"./tmp/{save_filename}_history.png"
df_scale = 1


def train_autoencoder(train_data, validation_data, test_data, dict_roles,
                      hero_cols, player_cols, match_cols,
                      n_heroes, epochs=100):

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
        training_df=train_data, validation_df=validation_data, epochs=100, best_model_filename=save_filename, verbose=False)
    autoencoder.save_loss_history(save_loss_history_path)
    print(
        f"Modelo salvo em {save_model_path} e histórico de perda salvo em {save_loss_history_path}.")
    print("Treinamento concluído.")
    return autoencoder, train_data, validation_data, test_data


def main():
    if (os.path.exists(dataset_full_path) and
            os.path.exists(dataset_metadata_path) and "--save" not in sys.argv):
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
        patches_info = ast.literal_eval(dataset_metadata["patches_info"].item())
        print("Dataset já carregado do arquivo.")
    else:
        dataset_name = "bwandowando/dota-2-pro-league-matches-2023"
        path = kagglehub.dataset_download(dataset_name)

        dataset, player_cols, match_cols, hero_cols = get_dataset(
            path, specific_patches=patches)
        train_data = dataset.sample(
            fraction=0.75 * df_scale, seed=42, shuffle=True)
        validation_data = dataset.sample(
            fraction=0.15 * df_scale, seed=42, shuffle=True)
        test_data = dataset.sample(
            fraction=0.15 * df_scale, seed=42, shuffle=True)

        heroes, _, _, dict_roles = get_heroes(path)
        n_heroes = heroes.select("hero_idx").count().collect().item()
        n_hero_stats = len(dict_roles) + len(hero_cols)
        n_player_stats = len(player_cols)

        patches_info = get_patches(path)

        if ("--save" in sys.argv):
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
            return

    if (os.path.exists(save_model_path) and "--force" not in sys.argv):
        print("Carregando modelo treinado...")
        autoencoder = Dota2Autoencoder.load_model(
            "./tmp/dota2_autoencoder.h5",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        print("Treinando novo modelo de autoencoder...")
        autoencoder, train_data, validation_data, test_data = train_autoencoder(
            train_data=dataset.sample(fraction=0.75, seed=42, shuffle=True),
            validation_data=dataset.sample(
                fraction=0.15, seed=42, shuffle=True),
            test_data=dataset.sample(fraction=0.15, seed=42, shuffle=True),
            dict_roles=dict_roles,
            hero_cols=hero_cols,
            player_cols=player_cols,
            match_cols=match_cols,
            n_heroes=n_heroes
        )

    train_data = dataset.sample(
        fraction=0.75 * df_scale, seed=42, shuffle=True)
    validation_data = dataset.sample(
        fraction=0.15 * df_scale, seed=42, shuffle=True)
    test_data = dataset.sample(
        fraction=0.15 * df_scale, seed=42, shuffle=True)

    if (test_data is not None):
        save_report(autoencoder, train_data,
                    validation_data, test_data,
                    patches_info, mse_threshold=0.1)


def save_report(
        autoencoder: Dota2Autoencoder, train_data: pl.DataFrame, validation_data: pl.DataFrame, test_data: pl.DataFrame,
        patches_info: dict[int, tuple[int, str]], mse_threshold: float):
    print("Salvando relatório de desempenho do modelo...")

    used_patches = [patch_name for patch_id, (patch_count, patch_name) in patches_info.items() if patch_id in patches]
    
    report_path = "./tmp/report.txt"
    lines = [
        "="*60,
        "RELATÓRIO DE DESEMPENHO DO MODELO DOTA 2",
        "="*60,
        "",
        "[DADOS DO CONJUNTO]",
        f"- Treinamento: {train_data.shape[0]} amostras",
        f"- Validação: {validation_data.shape[0]} amostras",
        f"- Teste: {test_data.shape[0]} amostras",
        f"- Threshold de MSE: {mse_threshold}",
        f"- Patches: {', '.join(str(patch) for patch in used_patches)}",
        "",
        "[ARQUIVOS GERADOS]",
        f"- Modelo salvo: {save_model_path}",
        f"- Histórico de perda: {save_loss_history_path}",
        f"- Plot de perda: {save_plot_loss_history_path}",
        "",
        "[ESTRUTURA DOS DADOS]",
        f"- Colunas de heróis: {', '.join(autoencoder.hero_columns)}",
        f"- Tamanho das colunas de heróis: {len(autoencoder.hero_columns)}",
        f"- Colunas de jogadores: {', '.join(autoencoder.player_columns)}",
        f"- Tamanho das colunas de jogadores: {len(autoencoder.player_columns)}",
        f"- Colunas de partidas: {', '.join(autoencoder.match_columns)}",
        f"- Tamanho das colunas de partidas: {len(autoencoder.match_columns)}",
        "",
        "[ARQUITETURA DO MODELO]",
        f"- Dimensão de embedding (picks de heróis): {autoencoder.hero_pick_embedding_dim}",
        f"- Dimensão de embedding (roles de heróis): {autoencoder.hero_role_embedding_dim}",
        f"- Dimensões de entrada: {autoencoder.input_dim}",
        f"- Dimensão latente: {autoencoder.latent_dim}",
        f"- Arquitetura: {([autoencoder.input_dim] + autoencoder.hidden_layers + [autoencoder.latent_dim] + list(reversed(autoencoder.hidden_layers)) + [autoencoder.input_dim])}",    
    ]    
    lines.append("")
    lines.append("[INFORMAÇÕES DO TREINAMENTO]")
    lines.append(f"- Épocas: {autoencoder.epoch_stop}")
    lines.append(f"- Perda de treino final: {autoencoder.avg_history[-1]:.6f}")
    lines.append(f"- Perda de validação final: {autoencoder.avg_val_history[-1]:.6f}")
    lines.append("")
    accuracy, avg_mse, min_mse, max_mse = autoencoder.test_model(test_data, threshold=mse_threshold)
    lines.append("[RESULTADOS NO CONJUNTO DE TESTE]")
    lines.append(f"- Acurácia: {accuracy:.2f} (Threshold {mse_threshold})")
    lines.append(f"- MSE médio: {avg_mse:.6f}")
    lines.append(f"- MSE mínimo: {min_mse:.6f}")
    lines.append(f"- MSE máximo: {max_mse:.6f}")
    lines.append("")
    lines.append("[INFORMAÇÕES DOS PATCHES]")
    for patch_id, (patch_count, patch_name) in patches_info.items():
        lines.append(f"- Patch {patch_id} ({patch_name}): {patch_count}")
    lines.append("")
    lines.append("[ROLES DISPONÍVEIS]")
    for role_name, role_id in autoencoder.dict_roles.items():
        lines.append(f"- {role_name}: {role_id}")

    lines.append("")
    lines.append("Teste do modelo concluído.")
    lines.append("="*60)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\r\n".join(lines))

    print(f"Relatório salvo em {report_path}.")
    print(f"Plotando histórico de perda em {save_plot_loss_history_path}...")
    plot_loss_history(save_loss_history_path, save_plot_loss_history_path,
                      title="Dota 2 Autoencoder Loss History")


if __name__ == "__main__":
    main()
