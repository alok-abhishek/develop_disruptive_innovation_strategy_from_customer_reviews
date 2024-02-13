import json
import torch
import os
from transformers import BertTokenizer, BertModel
from utils.sentiment_analysis.review_large_token_length import  token_length_review

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


base_path = "../../eda/kaggle/data"
input_filename = "us_airline_reviews_with_sentiment_analysis.json"
output_filename = "us_airline_reviews_sentiment_bert.json"
input_file = os.path.join(base_path, input_filename)
output_file = os.path.join(base_path, output_filename)


def get_bert_embeddings(review_text, model, tokenizer):
    # process the long review and reduce the size below 510 tokens for embedding using BERT
    adjusted_review = token_length_review(review_text)
    # Encode text
    inputs = tokenizer(
        adjusted_review,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()


def bert_embed_user_review(input_file, output_file):
    with open(input_file, "r") as file:
        data = json.load(file)

    # Generate embeddings
    for entry in data:
        entry["bert_vector"] = get_bert_embeddings(
            entry["Review"], model, tokenizer
        ).tolist()

    with open(output_file, "w") as file:
        json.dump(data, file, indent=4)

    print(f"BERT embeddings saved to {output_file}")


bert_embed_user_review(input_file, output_file)
