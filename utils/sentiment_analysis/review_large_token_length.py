from transformers import RobertaTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from summarization_bart_large_cnn import summarize_text
from split_and_summarize_recursively import chunk_and_summarize_very_large_review

# Load tokenizer
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer = AutoTokenizer.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)


def token_length_review(review):
    # convert to lower case for processing
    review = review.lower()
    # Tokenize the review
    tokens = tokenizer.encode(review, add_special_tokens=True)
    token_length = len(tokens)
    print("token_length: ", token_length)
    # print("tokens: ", tokens)

    # Check token length for bart summarization

    if token_length > 1000:
        # If review is too long, first chunk and summarize to get below 1000 tokens so that it can be processed using bart-large-cnn
        review = chunk_and_summarize_very_large_review(review)
        token_length = len(tokenizer.encode(review, add_special_tokens=True))

    # Check token length for roberta sentiment analysis

    if token_length > 510:
        # Summarize long reviews
        return summarize_text(review)
    else:
        # Return original review for shorter reviews
        return review

