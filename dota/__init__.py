import os
from .dota import Dota2
from .dataset import Dataset
from .logger import get_logger
import logger

os.makedirs("log", exist_ok=True)
os.makedirs("tmp", exist_ok=True)
os.makedirs("loss_history", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("best", exist_ok=True)


__all__ = ["Dota2", "Dataset", "get_logger", "logger"]