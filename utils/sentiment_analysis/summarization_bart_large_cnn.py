from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def summarize_text(article, max_length=500, min_length=400):
    # max length 500 to keep it less than bert's token limit, min length 400 to not summarize aggressively..
    # print("In summarize_text")
    # Summarize the article
    summarized = summarizer(
        article, max_length=max_length, min_length=min_length, do_sample=False
    )
    summary_article = summarized[0]["summary_text"]
    # print("summary_article after summarize_text:", summary_article)
    return summary_article
