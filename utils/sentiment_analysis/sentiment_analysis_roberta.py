import json
from transformers import pipeline
from review_large_token_length import token_length_review
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Load RoBERTa model for sentiment analysis
# classifier = pipeline('sentiment-analysis', model='roberta-base')
classifier = pipeline(
    "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)


def process_reviews_sentiment_analyzer(base_path, input_filename, output_filename):

    # Define a threshold for strongly positive or strongly negative as 66% confidence level - top 1/3 categorized as strong
    strong_threshold = 0.66

    # Read JSON file
    with open(f"{base_path}/{input_filename}", "r") as file:
        data = json.load(file)

    # Perform sentiment analysis and update data
    sentiment = []
    for airline, reviews in data.items():
        for review in reviews:
            adjusted_review = token_length_review(review["Review"])
            result = classifier(adjusted_review)
            # print("sentiment result: ", result)
            score = result[0]["score"]
            label = result[0]["label"]

            # Logic for enhance sentiment classification
            if label == "positive":
                if score > (strong_threshold):
                    sentiment = "strongly positive"
                else:
                    sentiment = "positive"
            elif label == "negative":
                if score > (strong_threshold):
                    sentiment = "strongly negative"
                else:
                    sentiment = "negative"
            else:
                sentiment = "neutral"

            # Add sentiment result to the review data
            review["Sentiment"] = sentiment
            # print("Sentiment Assigned: ", sentiment)

    # Write updated data to a new JSON file
    with open(f"{base_path}/{output_filename}", "w") as file:
        json.dump(data, file, indent=4)


base_path = "../../eda/kaggle/data"
input_filename = "airlines_with_30_plus_reviews_cleaned_1.json"
output_filename = "airlines_with_30_plus_reviews_cleaned_2.json"

process_reviews_sentiment_analyzer(base_path, input_filename, output_filename)
