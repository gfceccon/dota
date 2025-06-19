import os
import argparse

import kagglehub
from dota import Dota2
from dota import Dataset
import polars as pl
import random
import string

LEAGUES = f"Constants/Constants.Leagues.csv"
HEROES = f"Constants/Constants.Heroes.csv"
ITEMS = f"Constants/Constants.Items.csv"
PATCHES = f"Constants/Constants.Patch.csv"
REGION = f"Constants/Constants.Region.csv"


def METADATA(x): return f"{x}/main_metadata.csv"
def OBJECTIVES(x): return f"{x}/objectives.csv"
def PICKS_BANS(x): return f"{x}/picks_bans.csv"
def PLAYERS(x): return f"{x}/players.csv"
def EXP_ADV(x): return f"{x}/radiant_exp_adv.csv"
def GOLD_ADV(x): return f"{x}/radiant_gold_adv.csv"
def TEAM_FIGHTS(x): return f"{x}/teamfights.csv"


def try_load_cache(self, path: str = '') -> str:
    dataset_path = 'bwandowando/dota-2-pro-league-matches-2023'
    if (path == '' or path is None):
        path = os.path.expanduser(
            f'~/.cache/kagglehub/datasets/{dataset_path}/versions/')
        if (os.path.exists(path)):
            print(f"Checking cache path: {path}")
            paths = os.listdir(path)
            if (len(paths) > 0):
                path = os.path.join(path, paths[0])
                print(
                    f"Dataset path found in cache: {path}")
            else:
                print(
                    "Dataset path is not provided and cannot be found in KaggleHub.")
                path = ''
    if (path == '' or path is None):
        path = kagglehub.dataset_download(handle=self.dataset_name,)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Dota 2 Dataset Temporary Slice')
    parser.add_argument('--year', type=int, help='Ano entre (2021-2024)')
    parser.add_argument('--path', type=str,
                        help='Pasta do dataset caso já tenha sido baixado')
    args = parser.parse_args()

    if args.year:
        path = try_load_cache(args.path)

        def random_hash(length=16):
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        hash_str = random_hash()
        slice_path = os.path.join('tmp/slice', hash_str)

        os.makedirs(slice_path, exist_ok=True)
        os.makedirs(os.path.join(slice_path, 'Constants'), exist_ok=True)

        # Ensure all subdirectories exist before writing files

        def ensure_dir_for_file(file_path):
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)

        # Handle constants (no year argument)
        for const in [LEAGUES, HEROES, ITEMS, PATCHES]:
            out_path = os.path.join(slice_path, const)
            ensure_dir_for_file(out_path)
            pl.scan_csv(os.path.join(path, const)).head(
                10).collect().write_json(out_path + '.json')

        # Handle functions (require year argument)
        for func in [METADATA, OBJECTIVES, PICKS_BANS, PLAYERS, EXP_ADV, GOLD_ADV, TEAM_FIGHTS]:
            out_path = os.path.join(slice_path, func(args.year))
            ensure_dir_for_file(out_path)
            pl.scan_csv(os.path.join(path, func(args.year))).head(
                10).collect().write_json(out_path + '.json')
        print(f"Slice saved to {slice_path}")

    else:
        print("Nenhum ano especificado.")
        print(
            "Use o argumento --year para processar dados de um ano específico (2021-2024)")
