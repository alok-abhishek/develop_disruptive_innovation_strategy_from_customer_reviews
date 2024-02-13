import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Load data from JSON file
base_path = "../eda/kaggle/data"
input_filename = "us_airline_reviews_sentiment_bert.json"
input_file = os.path.join(base_path, input_filename)
output_filename = "us_airline_reviews_sentiment_bert_n_pca.json"
output_file = os.path.join(base_path, output_filename)

with open(input_file, "r") as file:
    data = json.load(file)

# Extract embeddings
embeddings = np.array([entry["bert_vector"] for entry in data])

# Standardize the embeddings
scaler = StandardScaler()
embeddings_standardized = scaler.fit_transform(embeddings)


# n_components_list = [0.85, 0.75, 0.65, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1]
n_components_list = [0.4, 0.3, 0.25, 0.2, 0.1]

for n_components in n_components_list:
    pca = PCA(n_components=n_components)
    embeddings_reduced = pca.fit_transform(embeddings_standardized)
    print(
        f"n_components = {n_components}, Reduced dimension: {embeddings_reduced.shape[1]}"
    )

    # Update JSON data with reduced embeddings
    for entry, reduced_embedding in zip(data, embeddings_reduced):
        entry[f"bert_pca_{n_components}"] = reduced_embedding.tolist()

# Save the updated data
with open(output_file, "w") as file:
    json.dump(data, file, indent=4)

print(f"Reduced embeddings with multiple PCA dimensions saved to {output_file}")
