from transformers import AutoTokenizer, pipeline, RobertaTokenizer
from summarization_bart_large_cnn import summarize_text

tokenizer = AutoTokenizer.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)


def chunk_and_summarize_very_large_review(very_large_review):

    tokens = tokenizer.encode(very_large_review, add_special_tokens=True)
    token_length = len(tokens)
    # print("token_length: ", token_length)

    chunk_size = 750  # Number of tokens in each chunk (in between 500 and 1000)
    overlap = 150  # Number of tokens for overlap (20%)
    chunks = []
    start_token = 0

    # Split the text into overlapping chunks
    while start_token < token_length:
        end_token = min(start_token + chunk_size, token_length)
        # print(
        #     "start: ", start_token, "end: ", end_token, "token_length: ", token_length
        # )
        chunk_tokens = tokens[start_token:end_token]
        # print("chunk_tokens: ", chunk_tokens)
        chunk = tokenizer.decode(chunk_tokens)
        # print("chunk: ", chunk)
        summarized_chunk = summarize_text(chunk)
        # print("summarized_chunk:", summarized_chunk)
        chunks.append(summarized_chunk)
        # print("chunks:", chunks)
        start_token = end_token - overlap if end_token < token_length else end_token

    # Concatenate all the summarized chunks into a final string
    final_review_summary = " ".join(chunks)
    # print("final_review_summary:", final_review_summary)
    final_review_summary_len = len(
        tokenizer.encode(final_review_summary, add_special_tokens=True)
    )
    # print("token length of final_review_summary: ", final_review_summary_len)

    # Recursive call if the summary is still too long
    if final_review_summary_len > 1000:
        # print("Recursive Call")
        return chunk_and_summarize_very_large_review(final_review_summary)

    return final_review_summary
