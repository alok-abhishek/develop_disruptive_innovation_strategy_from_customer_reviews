import numpy as np
import dotenv
import os
import json
from embed_reviews_bge_large_en import generate_embedding_bge_large_en
from embed_reviews_bge_small_en import generate_embedding_bge_small_en

# from embed_reviews_e5_mistral_7b_instruct import generate_embedding_e5_mistral

base_path = "../../eda/kaggle/data"
input_filename = "airline_reviews_with_sentiment_analysis.json"
output_filename_bge_large = "airline_reviews_sentiment_embed.json"
output_filename_bge_small = "airline_reviews_sentiment_bge_ls_embed.json"


def process_batch_bge_large(batch):

    for entry in batch:
        review_data = entry["Review"]
        user_review_embedding = generate_embedding_bge_large_en(review_data)
        entry["review_bge_embedding"] = user_review_embedding.tolist()
    return batch


def process_batch_bge_small(batch):

    for entry in batch:
        review_data = entry["Review"]
        user_review_embedding = generate_embedding_bge_small_en(review_data)
        entry["review_bge_small_embedding"] = user_review_embedding.tolist()
    return batch


def sentiment_bge_large_embed(input_filepath, output_filepath, batch_size=200):
    with open(input_filepath, "r") as file:
        data = json.load(file)

    # Open the output file in write mode
    with open(output_filepath, "w") as outfile:
        outfile.write("[")  # Start of JSON array

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            processed_batch = process_batch_bge_large(batch)

            # Write processed batch to file
            json_str = json.dumps(processed_batch, indent=4).strip("[]")
            if i + batch_size < len(data):
                json_str += ","  # Add comma for all but last batch
            outfile.write(json_str)

        outfile.write("]")  # End of JSON array


def sentiment_bge_small_embed(input_filepath, output_filepath, batch_size=100):
    with open(input_filepath, "r") as file:
        data = json.load(file)

    # Open the output file in write mode
    with open(output_filepath, "w") as outfile:
        outfile.write("[")  # Start of JSON array

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            processed_batch = process_batch_bge_small(batch)

            # Write processed batch to file
            json_str = json.dumps(processed_batch, indent=4).strip("[]")
            if i + batch_size < len(data):
                json_str += ","  # Add comma for all but last batch
            outfile.write(json_str)

        outfile.write("]")  # End of JSON array


"""
sentiment_bge_large_embed(
    os.path.join(base_path, input_filename),
    os.path.join(base_path, output_filename_bge_large),
)

"""

sentiment_bge_small_embed(
    os.path.join(base_path, output_filename_bge_large),
    os.path.join(base_path, output_filename_bge_small),
)
