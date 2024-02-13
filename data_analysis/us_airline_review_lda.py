import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel


nltk.download("stopwords")
nltk.download("punkt")

stop_words = set(stopwords.words("english"))

custom_stop_words = [
    "us",
    "get",
    "would",
    "could",
    "back",
    "got",
    "go",
    "use",
    "take",
    "went",
    "also",
]
stop_words.update(custom_stop_words)


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return [w for w in tokens if w.isalpha() and w not in stop_words]


def generate_lda_summary(df, sentiments):
    summary = ""

    for sentiment in sentiments:
        sentiment_df = df[df["Sentiment"] == sentiment]

        if sentiment_df.empty:
            summary += f"\nNo reviews for sentiment: {sentiment}\n"
            continue

        # Create dictionary and corpus for LDA
        dictionary = Dictionary(sentiment_df["processed_review"])
        corpus = [dictionary.doc2bow(text) for text in sentiment_df["processed_review"]]

        # LDA model
        num_topics = 6
        lda_model = LdaModel(
            corpus, num_topics=num_topics, id2word=dictionary, passes=20, random_state=0
        )

        # Append the topics for the sentiment to the summary
        summary += f"\nTopics for sentiment: {sentiment}\n"
        for idx, topic in lda_model.show_topics(
            formatted=True, num_topics=num_topics, num_words=20
        ):
            summary += f"Topic {idx}:\n{topic}\n"

    return summary.strip()


def generate_lda_summary_wrapper(input_file):
    # Load data
    with open(input_file, "r") as file:
        data = json.load(file)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Text Preprocessing

    df["processed_review"] = df["Review"].apply(preprocess_text)

    # Sentiment categories
    sentiments = [
        "strongly negative",
        "negative",
        "neutral",
        "positive",
        "strongly positive",
    ]

    lda_summary = generate_lda_summary(df, sentiments)

    return lda_summary
