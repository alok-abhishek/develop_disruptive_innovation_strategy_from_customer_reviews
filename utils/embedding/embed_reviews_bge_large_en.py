from sentence_transformers import SentenceTransformer
import json


# Load the Sentence Transformer model
model = SentenceTransformer("BAAI/bge-large-en-v1.5")


def generate_embedding_bge_large_en(review_data):
    # Create embeddings for the review data
    embeddings = model.encode(review_data, convert_to_tensor=False)

    return embeddings


# user_review = "Your user review text goes here."
# user_review_embedding = generate_embedding_bge_large_en(user_review)

# base_path = "../eda/kaggle/data"
# input_filename = "airline_reviews_with_sentiment_analysis.json"


# Load JSON data
# with open(f"{base_path}/{input_filename}", "r") as file:
#     data = json.load(file)

# review_text = data[0]["Review"]
# print("review_text: ", review_text)

# user_review_embedding = generate_embedding_bge_large_en(review_text)
# print("user_review_embedding: ", user_review_embedding)

# embeddings_bge = user_review_embedding[0]
# print("embeddings_bge: ", embeddings_bge)
