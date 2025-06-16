import os
import argparse
from dota import Dota2
from dota import Dataset
import polars as pl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dota 2 Dataset')
    parser.add_argument('--year', type=int, help='Ano entre (2020-2024)')
    parser.add_argument('--head', type=int, help='Salve apenas as primeiras N linhas do dataset', default=None)
    
    args = parser.parse_args()

    dataset = Dataset()
    dataset.save_metadata()
    
    if args.year:
        print(f"Processando ano {args.year}...")
        dataset.save_dataset(args.year, head=args.head)
    else:
        print("Nenhum ano especificado. Apenas os metadados foram salvos.")
        print("Use o argumento --year para processar dados de um ano específico (2020-2024)")