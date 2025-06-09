import itertools
import polars as pl
from dota import Dota2
from datetime import datetime

learning_rate = [0.001]
dropout = [0.3]
hidden_layers = [
    [128, 32],
    [256, 128],
    [128, 64, 32],
    [256, 128, 64],
]
hero_pick_embedding_size = [16, 8]
hero_role_embedding_size = [8, 4]
latent_dimensions = [2, 4]

configs = list(itertools.product(
    learning_rate,
    dropout,
    hidden_layers,
    hero_pick_embedding_size,
    hero_role_embedding_size,
    latent_dimensions
))


def run_permutations(_dota: Dota2, patch: list[int],
                     train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame,
                     epochs, model_filename, loss_filename, permutation_path):
    print(f"Available permutations: {len(configs)}")

    best_permutation = None
    best_permutation_loss = float('inf')
    best_permutation_mse = float('inf')
    best_permutation_accuracy = float('inf')
    results_filename = f"{permutation_path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    best_permutation_model_path = ""
    best_permutation_loss_path = ""
    with open(results_filename, "w") as f:
        f.write("Dota2 Autoencoder Permutation Results\n")
        f.write("=" * 40 + "\n")
    for idx, perm in enumerate(configs):
        print(
            f"Testing permutation {idx + 1}/{len(configs)}: {perm}")
        _dota.set_config({
            "learning_rate": perm[0],
            "dropout": perm[1],
            "hidden_layers": perm[2],
            "hero_pick_embedding_size": perm[3],
            "hero_role_embedding_size": perm[4],
            "latent_dimensions": perm[5],
        }, True)
        loss_path = f"{loss_filename}_{idx + 1}.csv"
        model_path = f"{model_filename}_{idx + 1}.h5"
        autoencoder, loss = _dota.train_or_load_autoencoder(
            train_df, val_df, test_df, model_path, loss_path, epochs=epochs, silent=True)
        autoencoder.save_model(model_path + f"_{idx + 1}.h5", True)
        accuracy, avg_mse, _, _ = _dota.test_autoencoder(
            test_df, 0.1, silent=True)

        if loss < best_permutation_loss:
            best_permutation_loss = loss
            best_permutation = perm
            best_permutation_accuracy = accuracy
            best_permutation_mse = avg_mse
            best_permutation_loss_path = loss_path
            best_permutation_model_path = model_path

        print(
            f"Permutation {perm}, Loss: {loss}, Accuracy: {accuracy}, Avg MSE: {avg_mse}")
        with open(results_filename, "a") as f:
            f.write(
                f"Permutation {idx + 1}/{len(configs)}: {perm}, Loss: {best_permutation_loss}\n")
            f.write(
                f"Accuracy: {accuracy}, Avg MSE: {avg_mse}")
            f.write(f"Model saved to: {model_path}_{idx + 1}.h5\n")
            f.write("-" * 40 + "\n")
    print(
        f"Best permutation: {best_permutation} with loss {best_permutation_loss}")
    print(
        f"Test Accuracy: {best_permutation_accuracy}, Avg MSE: {best_permutation_mse}")
    plot_path = f"{model_filename}_plot.png"
    report_path = f"{permutation_path}_report.txt"
    try:
        _dota.save_report(train_df, val_df, test_df, patch, 0.01,
                          report_path, best_permutation_loss_path, plot_path)
    except Exception as e:
        print(f"Error saving report: {e}")
    return best_permutation, best_permutation_loss_path, best_permutation_model_path, plot_path, report_path
