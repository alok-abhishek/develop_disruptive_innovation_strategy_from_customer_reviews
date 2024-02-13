import json
from sklearn.feature_extraction.text import TfidfVectorizer
import os

base_path = "../../eda/kaggle/data"
input_filename = "us_airline_reviews_with_sentiment_analysis.json"
output_filename = "us_airline_reviews_with_sentiment_analysis_tfidf.json"
input_file = os.path.join(base_path, input_filename)
output_file = os.path.join(base_path, output_filename)

with open(input_file, "r") as file:
    data = json.load(file)

# Extract reviews
reviews = [entry["Review"] for entry in data]

# Apply TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english", max_df=0.85)
tfidf_matrix = vectorizer.fit_transform(reviews)

# Convert the sparse TF-IDF matrix to a dense array
tfidf_dense = tfidf_matrix.toarray()

# Add dense TF-IDF vectors to data
for entry, tfidf_vector in zip(data, tfidf_dense):
    entry["TFIDF"] = tfidf_vector.tolist()

# Save the updated data to a new file
with open(output_file, "w") as file:
    json.dump(data, file, indent=4)

print(f"TF-IDF vectors saved to {output_file}")
