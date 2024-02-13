import json
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel


def preprocess_text(text, stop_words):
    tokens = word_tokenize(text.lower())
    return [w for w in tokens if w.isalpha() and not w in stop_words]


def generate_lda_summary(df, sentiments):
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

    df["processed_review"] = df["Review"].apply(
        lambda x: preprocess_text(x, stop_words)
    )

    for sentiment in sentiments:
        sentiment_df = df[df["Sentiment"] == sentiment]

        if sentiment_df.empty:
            print(f"\nNo reviews for sentiment: {sentiment}")
            continue

        processed_docs = sentiment_df["processed_review"].tolist()
        dictionary = Dictionary(processed_docs)
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        coherence_values = []
        model_list = []
        topic_numbers = range(2, 6)

        for num_topics in topic_numbers:
            model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
            model_list.append(model)
            coherencemodel = CoherenceModel(
                model=model,
                texts=processed_docs,
                dictionary=dictionary,
                coherence="c_v",
            )
            coherence = coherencemodel.get_coherence()
            coherence_values.append(coherence)

        optimal_num_topics = topic_numbers[
            coherence_values.index(max(coherence_values))
        ]
        optimal_model = model_list[coherence_values.index(max(coherence_values))]

        print(
            f"\nOptimal Number of Topics for '{sentiment}': {optimal_num_topics}, Maximum Coherence Score: {max(coherence_values):.4f}"
        )

        # Print the topics for the sentiment
        for idx, topic in optimal_model.show_topics(
            num_topics=optimal_num_topics, num_words=10
        ):
            print(f"Sentiment: {sentiment}, Topic {idx}:\n{topic}\n")


def generate_lda_summary_wrapper(input_file):
    with open(input_file, "r") as file:
        data = json.load(file)

    df = pd.DataFrame(data)
    sentiments = [
        "strongly negative",
        "negative",
        "neutral",
        "positive",
        "strongly positive",
    ]

    generate_lda_summary(df, sentiments)


if __name__ == "__main__":
    nltk.download("stopwords")
    nltk.download("punkt")

    base_path = "../eda/kaggle/data"
    input_filename = "us_airline_reviews_with_sentiment_analysis.json"
    input_file = os.path.join(base_path, input_filename)

    generate_lda_summary_wrapper(input_file)
