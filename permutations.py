import itertools
import polars as pl
from dota import Dota2
from datetime import datetime
from model import Dota2Autoencoder
import hashlib
import os
import shutil

learning_rate = [0.001]
dropout = [0.3]
hidden_layers = [
    [64, 32],
    [64, 32, 16],
    [128, 64],
    [128, 64, 32],
    [256, 128],
    [256, 128, 64],
    [256, 128, 64, 32],
]
hero_pick_embedding_size = [8]
hero_role_embedding_size = [4]
latent_dimensions = [2, 3, 4, 8]

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
                     model_filename, loss_filename, permutation_filename):
    print(f"Available permutations: {len(configs)}")

    def configs_hash(configs):
        configs_str = (str(_dota.patches) + str(configs)).encode("utf-8")
        return hashlib.sha256(configs_str).hexdigest()

    hash = configs_hash(configs)
    os.makedirs(f"./tmp/{hash}", exist_ok=True)
    print(f"Permutations hash: {hash}")

    permutation_path = f"./tmp/{hash}/{permutation_filename}"
    permutation_models: list[tuple[str, float,
                                   float, float, int, str, str]] = []

    if os.path.exists(f"{permutation_path}.csv"):
        print(f"Permutation results already exist: {permutation_path}.csv")
        df = pl.read_csv(f"{permutation_path}.csv")
        best = df.sort("loss").head(1).to_dicts()[0]
        print(
            f"Best permutation: {best['permutation']} with loss {best['loss']}")
        print(
            f"Test Accuracy: {best['accuracy']}, Avg MSE: {best['avg_mse']}, Stopped at: {best['stopped']}")
        return df, best["permutation"], best["loss_path"], best["model_path"], None, None

    with open(f"{permutation_path}.txt", "w") as f:
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
        loss_path = f"./tmp/{hash}/{loss_filename}_{perm_str}.csv"
        model_path = f"./tmp/{hash}/{model_filename}_{perm_str}.h5"

        autoencoder = _dota.create_autoencoder()
        autoencoder.train_data(
            train_df, val_df, best_model_filename=model_path, epochs=epochs, silent=True)
        autoencoder.save_loss_history(loss_path, silent=True)
        accuracy, avg_mse, _, _ = autoencoder.test_model(test_df)
        loss = autoencoder.best_val_loss

        permutation_models.append(
            (perm_str, loss, accuracy, avg_mse, autoencoder.train_stopped, model_path, loss_path))

        print(
            f"Permutation {perm}, Loss: {loss}, Accuracy: {accuracy}, Avg MSE: {avg_mse} Epochs: {autoencoder.train_stopped}")
        with open(f"{permutation_path}.txt", "a") as f:
            f.write(
                f"Permutation {idx + 1}/{len(configs)}: {perm_str}, Loss: {loss}\n")
            f.write(
                f"Accuracy: {accuracy}, Avg MSE: {avg_mse}\n")
            f.write(f"Model saved to: {model_path}\n")
            f.write("-" * 40 + "\n")

    df = pl.DataFrame(permutation_models, strict=False).transpose()
    df.columns = ["permutation", "loss", "accuracy",
                  "avg_mse", "stopped", "model_path", "loss_path"]
    df.write_csv(f"{permutation_path}.csv")

    best = df.sort("loss").head(1).to_dicts()[0]

    print(
        f"Best permutation: {best["permutation"]} with loss {best["loss"]}")
    print(
        f"Test Accuracy: {best["accuracy"]}, Avg MSE: {best["avg_mse"]}, Stopped at: {best["stopped"]}")

    plot_path = f"./tmp/{hash}/{model_filename}_plot.png"
    report_path = f"./tmp/{hash}/{permutation_filename}_report.txt"

    write_report(best, report_path)

    shutil.copyfile(best["model_path"], "best.h5")
    return df, best["permutation"], best["loss_path"], best["model_path"], plot_path, report_path


def write_report(best, report_path):
    with open(report_path, "w") as f:
        f.write("Dota2 Autoencoder Permutation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Best permutation: {best['permutation']}\n")
        f.write(f"Loss: {best['loss']}\n")
        f.write(f"Accuracy: {best['accuracy']}\n")
        f.write(f"Avg MSE: {best['avg_mse']}\n")
        f.write(f"Stopped at: {best['stopped']}\n")
        f.write(f"Model path: {best['model_path']}\n")
        f.write(f"Loss path: {best['loss_path']}\n")


if __name__ == "__main__":
    dota = Dota2([56])
    tr, vl, te = dota.prepare_data_splits(dota.dataset)
    run_permutations(
        dota, 100,
        tr, vl, te,
        model_filename="dota_autoencoder",
        loss_filename="dota_autoencoder_loss",
        permutation_filename="dota_autoencoder_permutations"
    )
