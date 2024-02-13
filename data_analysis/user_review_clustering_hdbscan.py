import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN
import os

base_path = "../eda/kaggle/data"
input_filename = "us_airline_reviews_sentiment_bert_n_pca.json"
input_file = os.path.join(base_path, input_filename)

# Load the data
with open(input_file, "r") as file:
    data = json.load(file)

# Extract embeddings into a NumPy array
embeddings = np.array([entry["bert_pca_0.4"] for entry in data])

# Print details of embedding vector

print("Shape of embeddings:", embeddings.shape)
print("Type of embeddings:", type(embeddings))
print("Size of embeddings (number of elements):", embeddings.size)
print("Sample of embeddings (first 5x5 block):\n", embeddings[:5, :5])


# Compute the cosine distance matrix
cosine_dist_matrix = squareform(pdist(embeddings, "cosine"))

# Print details of cosine_dist_matrix
print("Shape of cosine_dist_matrix:", cosine_dist_matrix.shape)
print("Type of matrix:", type(cosine_dist_matrix))
print("Size of matrix (number of elements):", cosine_dist_matrix.size)
print("Sample of matrix (first 5x5 block):\n", cosine_dist_matrix[:5, :5])

# Set the diagonal elements to zero
np.fill_diagonal(cosine_dist_matrix, 0)

# Print details of cosine_dist_matrix
print("Shape of cosine_dist_matrix:", cosine_dist_matrix.shape)
print("Type of matrix:", type(cosine_dist_matrix))
print("Size of matrix (number of elements):", cosine_dist_matrix.size)
print("Sample of matrix (first 5x5 block):\n", cosine_dist_matrix[:5, :5])


best_score = -1
best_min_cluster_size = None
best_labels = None


for min_cluster_size in range(3, 10):
    hdbscan_cluster = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = hdbscan_cluster.fit_predict(embeddings)

    # Compute silhouette score
    if len(set(labels)) > 1 and -1 not in labels:
        score = silhouette_score(embeddings, labels)
        print(f"Min Cluster Size: {min_cluster_size}, Silhouette Score: {score:.2f}")
    else:
        print(
            f"Min Cluster Size: {min_cluster_size}, Silhouette Score: Not applicable (single cluster or noise)"
        )

"""

for min_cluster_size in range(3, 10):
    hdbscan_cluster = HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed")
    labels = hdbscan_cluster.fit_predict(cosine_dist_matrix)

    # Compute silhouette score
    if len(set(labels)) > 1 and -1 not in labels:
        score = silhouette_score(embeddings, labels)
        print(f"Min Cluster Size: {min_cluster_size}, Silhouette Score: {score:.2f}")
    else:
        print(
            f"Min Cluster Size: {min_cluster_size}, Silhouette Score: Not applicable (single cluster or noise)"
        )
"""
