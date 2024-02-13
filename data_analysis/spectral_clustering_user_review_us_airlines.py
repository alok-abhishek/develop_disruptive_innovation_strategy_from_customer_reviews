import json
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import os

# Load data from JSON file
base_path = "../eda/kaggle/data"
input_filename = "us_airline_reviews_sentiment_bert_n_pca.json"
input_file = os.path.join(base_path, input_filename)

with open(input_file, "r") as file:
    data = json.load(file)

pca_key = "bert_pca_0.1"
embeddings = np.array([entry[pca_key] for entry in data])

# Loop through different numbers of clusters
for n_clusters in range(2, 7):
    sc = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors")
    labels = sc.fit_predict(embeddings)

    # Compute silhouette score
    silhouette_avg = silhouette_score(embeddings, labels)
    print(f"For n_clusters = {n_clusters}, Silhouette Score: {silhouette_avg:.2f}")

