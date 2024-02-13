from sentence_transformers import SentenceTransformer
import json


# Load the Sentence Transformer model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def generate_embedding_bge_small_en(review_data):
    # Create embeddings for the review data
    embeddings = model.encode(review_data, convert_to_tensor=False)

    return embeddings
