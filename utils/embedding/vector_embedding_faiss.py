import json
from transformers import pipeline
import faiss
import numpy as np
from embed_reviews_bge_large_en import generate_embedding_bge_large_en
from embed_reviews_e5_mistral_7b_instruct import generate_embedding_e5_mistral


base_path = "../../eda/kaggle/data"
input_filename = "airline_reviews_with_sentiment_analysis.json"


# Load JSON data
with open(f"{base_path}/{input_filename}", "r") as file:
    data = json.load(file)


def create_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index


# Generate embeddings
for review in data:
    review_text = review["Review"]
    review["Embedding_Bge"] = generate_embedding_bge_large_en(review_text)


# Extracting embeddings for FAISS
embeddings_bge = [review["Embedding_Bge"] for review in data]

# Create FAISS indices
index_bge = create_faiss_index(embeddings_bge)
