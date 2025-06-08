import sys
import ast
from plot import plot_loss_history
from model import Dota2Autoencoder
from dota import get_filenames, load_or_prepare_dataset, train_or_load_autoencoder, save_dataset_and_metadata, prepare_data_splits
import polars as pl


def parse_args():
    patches = [53]
    mse_threshold = 0.1
    for arg in sys.argv:
        if arg.startswith('--patches='):
            value = arg.split('=', 1)[1]
            try:
                patches_val = ast.literal_eval(value)
                if isinstance(patches_val, list):
                    patches = patches_val
            except Exception:
                pass
        if arg.startswith('--mse_threshold='):
            value = arg.split('=', 1)[1]
            try:
                mse_threshold = float(value)
            except Exception:
                pass
    return patches, mse_threshold


def save_report(
        autoencoder: Dota2Autoencoder, train_data: pl.DataFrame, validation_data: pl.DataFrame, test_data: pl.DataFrame,
        patches: list[int], patches_info: dict[int, tuple[int, str]], mse_threshold: float,
        save_model_path: str, save_loss_history_path: str, save_plot_loss_history_path: str, report_path: str):
    print("Salvando relatório de desempenho do modelo...")

    used_patches = [patch_name for patch_id,
                    (_, patch_name) in patches_info.items() if patch_id in patches]

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
    lines.append(
        f"- Perda de validação final: {autoencoder.avg_val_history[-1]:.6f}")
    lines.append("")
    accuracy, avg_mse, min_mse, max_mse = autoencoder.test_model(
        test_data, threshold=mse_threshold)
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


def main():
    patches, mse_threshold = parse_args()
    save_filenames = get_filenames(patches)
    dataset, player_cols, match_cols, hero_cols, dict_roles, n_heroes, n_hero_stats, n_player_stats, patches_info, is_new = load_or_prepare_dataset(
        save_filenames["dataset_full_path"],                                                                                      save_filenames["dataset_metadata_path"], patches)
    if is_new and ("--save" in sys.argv):
        save_dataset_and_metadata(dataset, player_cols, match_cols, hero_cols,
                                  dict_roles, n_heroes, n_hero_stats, n_player_stats, patches_info,
                                  save_filenames["dataset_full_path"], save_filenames["dataset_metadata_path"])
        return
    autoencoder, train_data, validation_data, test_data = train_or_load_autoencoder(
        dataset, dict_roles, hero_cols, player_cols, match_cols, n_heroes,
        save_filenames["save_model_path"], save_filenames["save_loss_history_path"], train=True if "--train" in sys.argv else False)
    if (train_data is None or validation_data is None or test_data is None):
        train_data, validation_data, test_data = prepare_data_splits(
            dataset, 1.0)
    save_report(autoencoder, train_data, validation_data, test_data, patches, patches_info, mse_threshold,
                save_filenames["save_model_path"], save_filenames["save_loss_history_path"], save_filenames["save_plot_loss_history_path"], save_filenames["report_path"])


if __name__ == "__main__":
    main()
