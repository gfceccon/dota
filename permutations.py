import itertools
import polars as pl
from dota import Dota2
from datetime import datetime
from model import Dota2Autoencoder

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


def run_permutations(_dota: Dota2, epochs: int,
                     train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame,
                     model_filename, loss_filename, permutation_path):
    print(f"Available permutations: {len(configs)}")

    results_filename = f"./tmp/{permutation_path}_{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"
    permutation_models: list[tuple[str, float,
                                   float, float, int, str, str]] = []

    with open(f"{results_filename}.txt", "w") as f:
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
        perm_str = "_".join([str(p) for p in perm]).replace(
            ".", "_").replace(" ", "_").replace(",", "_")
        perm_str = "".join(c for c in perm_str if c.isalnum() or c == "_")
        loss_path = f"./loss/{loss_filename}_{perm_str}.csv"
        model_path = f"./best/{model_filename}_{perm_str}.h5"

        autoencoder = _dota.create_autoencoder()
        autoencoder.train_data(
            train_df, val_df, best_model_filename=model_path, epochs=epochs, silent=True)
        autoencoder.save_loss_history(loss_path, silent=True)
        accuracy, avg_mse, _, _ = autoencoder.test_model(test_df)
        loss = autoencoder.best_val_loss

        permutation_models.append(
            (perm_str, loss, accuracy, avg_mse, autoencoder.train_stopped, model_path, loss_path))

        print(
            f"Permutation {perm_str}, Loss: {loss}, Accuracy: {accuracy}, Avg MSE: {avg_mse} Epochs: {autoencoder.train_stopped}")
        with open(f"{results_filename}.txt", "a") as f:
            f.write(
                f"Permutation {idx + 1}/{len(configs)}: {perm_str}, Loss: {loss}\n")
            f.write(
                f"Accuracy: {accuracy}, Avg MSE: {avg_mse}\n")
            f.write(f"Model saved to: {model_path}\n")
            f.write("-" * 40 + "\n")

    df = pl.DataFrame(permutation_models, strict=False).transpose()
    df.columns = ["permutation", "loss", "accuracy",
                  "avg_mse", "stopped", "model_path", "loss_path"]
    df.write_csv(f"{results_filename}.csv")

    best = df.sort("loss").head(1).to_dicts()[0]

    print(
        f"Best permutation: {best["permutation"]} with loss {best["loss"]}")
    print(
        f"Test Accuracy: {best["accuracy"]}, Avg MSE: {best["avg_mse"]}, Stopped at: {best["stopped"]}")

    plot_path = f"./plot/{model_filename}_plot.png"
    report_path = f"./report/{permutation_path}_report.txt"
    try:
        _dota.save_report(train_df, val_df, test_df, report_path,
                          best["loss_path"], plot_path)
    except Exception as e:
        print(f"Error saving report: {e}")
    return best, best["loss_path"], best["model_path"], plot_path, report_path


if __name__ == "__main__":
    dota = Dota2([55])
    tr, vl, te = dota.prepare_data_splits(dota.dataset)
    run_permutations(
        dota, 200,
        tr, vl, te,
        model_filename="dota_autoencoder",
        loss_filename="dota_autoencoder_loss",
        permutation_path="dota_autoencoder_permutations"
    )
