import os
import argparse
from dota import Dota2
from dota import OptimizedDataset
import pandas as pd
from dota.logger import get_logger
log = get_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dota 2 Dataset')
    parser.add_argument('--year', type=int, help='Ano entre (2021-2024)')
    args = parser.parse_args()
    
    if args.year:
        year=args.year
        ds = OptimizedDataset()
        players = ds.players(year=year).head(5).collect()
        log.info(f"{players}")
        players.write_json(f"players_{year}.json")
    else:
        log.error("Use o argumento --year para processar dados de um ano específico (2021-2024)")