import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import json

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-mistral-7b-instruct")
model = AutoModel.from_pretrained("intfloat/e5-mistral-7b-instruct")


def mean_pooling(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    # Create an attention mask for the embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    )

    # Sum embeddings and attention mask
    sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)

    # Sum the attention mask
    sum_mask = input_mask_expanded.sum(1)

    # Avoid division by 0 and perform mean pooling
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    mean_pooled_embeddings = sum_embeddings / sum_mask

    return mean_pooled_embeddings


def generate_embedding_e5_mistral(user_review: str) -> torch.Tensor:

    # Tokenize the user review
    encoded_input = tokenizer(
        user_review, return_tensors="pt", max_length=4096, truncation=True, padding=True
    )

    # Generate embeddings using the model
    with torch.no_grad():
        outputs = model(**encoded_input)

    # Extract and normalize the embedding of the last token
    embedding = mean_pooling(outputs.last_hidden_state, encoded_input["attention_mask"])
    normalized_embedding = F.normalize(embedding, p=2, dim=1)

    return normalized_embedding


# user_review = "Your user review text goes here."
# user_review_embedding = generate_embedding_e5_mistral(user_review)

# base_path = "../eda/kaggle/data"
# input_filename = "airline_reviews_with_sentiment_analysis.json"


# Load JSON data
# with open(f"{base_path}/{input_filename}", "r") as file:
#     data = json.load(file)

# review_text = data[0]["Review"]
# print("review_text: ", review_text)

# user_review_embedding = generate_embedding_e5_mistral(review_text)
# print("user_review_embedding: ", user_review_embedding)

# embeddings_mistral = user_review_embedding[0]
# print("embeddings_mistral: ", embeddings_mistral)
