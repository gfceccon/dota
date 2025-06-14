import os
from dota.logger import get_logger, LogLevel
from dota.dataset.dataset import Dataset

os.makedirs("log", exist_ok=True)
os.makedirs("tmp", exist_ok=True)
os.makedirs("loss_history", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("best", exist_ok=True)


log = get_logger("Dota2", LogLevel.INFO)

class Dota2():
    def __init__(self, dataset: Dataset):
        ...