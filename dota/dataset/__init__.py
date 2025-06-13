import kagglehub
from .dataset import Dataset

DATASET_NAME = "bwandowando/dota-2-pro-league-matches-2023"
PATH = kagglehub.dataset_download(DATASET_NAME)

__all__ = ["Dataset"]