import sys
import ast
from plot import plot_loss_history
from model import Dota2Autoencoder
from dota import Dota2
import polars as pl


def parse_args():
    patches = []
    for arg in sys.argv:
        if arg.startswith('--patches='):
            value = arg.split('=', 1)[1]
            try:
                patches_val = ast.literal_eval(value)
                if isinstance(patches_val, list):
                    patches = patches_val
            except Exception:
                pass
    return patches


def main():
    patches = parse_args()
    dota = Dota2(patches)

    dataset, _ = dota.load_or_prepare_dataset("./tmp/dota2.json", "./tmp/dota2_metadata.json",)
    dota.save_dataset_and_metadata(dataset, "./tmp/dota2.json", "./tmp/dota2_metadata.json")
    train, val, test = dota.prepare_data_splits(dataset, 1)
    dota.train_autoencoder(100)
    dota.save_report(train, val, test, "./tmp/dota2_autoencoder_main.h5",
                     "./tmp/dota2_autoencoder_main.csv", "./tmp/dota2_plot.png")


if __name__ == "__main__":
    main()
