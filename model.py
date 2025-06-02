import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class Dota2Autoencoder(nn.Module):
    def __init__(
        self,
        n_heroes: int,
        hero_embedding_dim: int,
        n_stats: int,
        stats_embedding_dim: int
    ):
        super(Dota2Autoencoder, self).__init__()
