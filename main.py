import os
import argparse
from dota import Dota2
from dota import Dataset
import polars as pl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dota 2 Dataset')
    parser.add_argument('--year', type=int, help='Ano entre (2021-2024)')
    args = parser.parse_args()
    
    if args.year:
        print(f"Carregando ano {args.year}...")
        dataset_lf = Dataset.load(os.getcwd(), args.year)
        dataset = Dataset(
            dataset=dataset_lf
        )
        lf = dataset.data
        if lf is not None:
            print(lf.head().collect())
    else:
        print("Use o argumento --year para processar dados de um ano específico (2021-2024)")