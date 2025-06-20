import os
import argparse
from dota import Dota2
from dota.logger import get_logger
from dota import Dataset
log = get_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dota 2 Dataset')
    parser.add_argument('--year', type=int, help='Ano entre (2021-2024)', required=True)
    parser.add_argument('--slice', action='store_true', help='Criar um slice temporário do dataset')
    parser.add_argument('--download', action='store_true', help='Força baixar o dataset completo')
    args = parser.parse_args()

    if args.year < 2021 or args.year > 2024:
        log.error("Ano inválido. Deve ser entre 2021 e 2024.")
        exit(1)
    if args.download:
        log.info(f"Baixando o dataset de Dota 2...")
        Dataset.force_download()
    if args.slice:
        log.info(f"Criando um slice temporário do dataset de Dota 2 para o ano {args.year}...")
        ds = Dataset(args.year)
        ds.get(slice=True)
    