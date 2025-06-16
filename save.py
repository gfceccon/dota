import os
import argparse
from dota import Dota2
from dota import Dataset
import polars as pl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dota 2 Dataset')
    parser.add_argument('--year', type=int, help='Ano entre (2021-2024)')
    parser.add_argument('--ti', type=bool, help='Salve dados do The International', default=False)
    parser.add_argument('--sample', type=int, help='Sample de dados', default=0)
    
    args = parser.parse_args()

    dataset = Dataset()
    dataset.save_metadata()
    
    if args.year:
        print(f"Processando ano {args.year}...")
        dataset.save_dataset(args.year, args.sample, ti=args.ti)
    else:
        print("Nenhum ano especificado. Apenas os metadados foram salvos.")
        print("Use o argumento --year para processar dados de um ano específico (2021-2024)")