import random
import numpy as np
import torch
from model import Dota2Autoencoder
from sklearn.cluster import KMeans
import sklearn as sk
import polars as pl


class Dota2Clustering:
    def __init__(self, autoencoder: Dota2Autoencoder):
        self.autoencoder = autoencoder

    def cluster(self, dataset: pl.DataFrame, n_clusters: int = 5):
        self.seed = 42  # Set a random state for reproducibility
        self.batch_size = 64  # Define the batch size for processing
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        pl.set_random_seed(self.seed)
        sk.random.seed(self.seed)
        
        encoded = []
        similarity = torch.Tensor([])
        # Encode the dataset to get the latent space representation
        for row in dataset.iter_rows(named=True):
            data_np = np.array(row.items())
            encoded_data, reconstructed = self.autoencoder.encode(data_np, 1, dataset.columns)
            loss = torch.cosine_similarity(encoded_data, reconstructed)
            similarity = torch.cat((similarity, loss))
            encoded_list = encoded_data.tolist()
            encoded.extend(encoded_list)
            
        similarity_np = similarity.numpy()
        print(f"Similarity scores sum: {similarity_np.sum()}")
        # Perform clustering on the encoded data
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(np.array(encoded))
            
            
        print(f"Clustering completed with {n_clusters} clusters.")
            
        return kmeans, cluster_labels, encoded
