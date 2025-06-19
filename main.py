import os
import argparse
from dota import OptimizedDataset
from dota.logger import get_logger
log = get_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dota 2 Dataset')
    parser.add_argument('--year', type=int, help='Ano entre (2021-2024)')
    args = parser.parse_args()
    
    if args.year and args.year in [2021, 2022, 2023, 2024]:
        year=args.year
        log.info(f"Processando dados do ano {year}")
        ds = OptimizedDataset()
        cache = ds.get(year)
        log.info(f"Dados do ano {year} processados com sucesso.")
        log.info(f"Cache criado: {cache}")
    else:
        log.error("Use o argumento --year para processar dados de um ano específico (2021-2024)")