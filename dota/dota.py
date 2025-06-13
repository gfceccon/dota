from logger import get_logger, LogLevel
from dataset import Dataset


log = get_logger("Dota2", LogLevel.INFO)

class Dota2():
    def __init__(self, dataset: Dataset):
        ...