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
        print(f"Processando ano {args.year}...")
        df = Dataset().save_dataset(args.year)
        print("Dados salvos com sucesso!")
        print("Exibindo as primeiras linhas do DataFrame:")
        print(df.head())
        print("Formato do DataFrame:")
        print(df.shape)

    else:
        print("Nenhum ano especificado.")
        print("Use o argumento --year para processar dados de um ano específico (2021-2024)")