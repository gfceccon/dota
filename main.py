import sys
import ast
from plot import plot_loss_history
from model import Dota2Autoencoder
from dota import Dota2
import polars as pl


def parse_args():
    patches = []
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


def main():
    patches, mse_threshold = parse_args()
    dota = Dota2(patches)

    dataset, _ = dota.load_or_prepare_dataset(
        "./tmp/dota2.json", "./tmp/dota2_metadata.json",)
    train, val, test = dota.prepare_data_splits(dataset, 0.8)
    dota.train_autoencoder(
        train, val, test, "./tmp/dota2_autoencoder_main.h5", "./tmp/dota2_autoencoder_main.csv")
    dota.save_report(train, val, test, patches, mse_threshold, "./tmp/dota2_autoencoder_main.h5",
                     "./tmp/dota2_autoencoder_main.csv", "./tmp/dota2_plot.png")


if __name__ == "__main__":
    main()
