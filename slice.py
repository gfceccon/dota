import os
import argparse

import kagglehub
from dota import Dota2
from dota import Dataset
import polars as pl
import random
import string

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Dota 2 Dataset Temporary Slice')
    parser.add_argument('--year', type=int, help='Ano entre (2021-2024)')
    args = parser.parse_args()

    if args.year:
        def random_hash(length=16):
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        hash_str = random_hash()
        slice_path = os.path.join('tmp/slice', hash_str)

        dataset = Dataset(year=2023)
        data = dataset.get()

        if not os.path.exists(slice_path):
            os.makedirs(slice_path)
            data.head(10).collect().write_json(
                os.path.join(slice_path, 'head.json'))

    else:
        print("Nenhum ano especificado.")
        print(
            "Use o argumento --year para processar dados de um ano específico (2021-2024)")
