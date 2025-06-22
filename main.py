import os
import argparse
from dota import Dota2
from dota.logger import get_logger
from dota import Dota2Dataset
import polars as pl
log = get_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dota 2 Dataset')
    parser.add_argument('--year', type=int,
                        help='Ano entre (2021-2024)', required=True)
    parser.add_argument('--download', action='store_true',
                        help='Força baixar o dataset completo')
    args = parser.parse_args()

    if args.year < 2021 or args.year > 2024:
        log.error("Ano inválido. Deve ser entre 2021 e 2024.")
        exit(1)

    force_download = False
    if args.download:
        force_download = True

    log.info(f"Iniciando Dota 2 Dataset para o ano {args.year}...")
    dataset = Dota2Dataset(year=args.year, force_download=force_download)
    data_path = dataset.get()
    if not os.path.exists(data_path):
        log.error(
            f"Erro ao carregar o dataset do ano {args.year}. Verifique se o caminho está correto.")
        exit(1)

    log.info(f"Dataset do ano {args.year} carregado com sucesso.")
    data = pl.scan_parquet(data_path, )
    schema = data.collect_schema()
    head = data.head(5).collect()
    log.info(f"Schema do dataset:\n{schema.names()}")
    log.info(f"Primeiras 5 linhas do dataset:\n{head}")
    head.write_json("head.json", )
