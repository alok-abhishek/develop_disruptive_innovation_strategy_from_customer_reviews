import json

# Load JSON data
with open("./kaggle/data/airlines_with_30_plus_reviews_cleaned_2.json", "r") as file:
    data = json.load(file)

# Flatten the JSON structure
flattened_data = []
for airline, reviews in data.items():
    for review in reviews:
        # Create a new dict with Airline first
        flattened_review = {"Airline": airline}
        # Update this dict with the review details
        flattened_review.update(review)
        flattened_data.append(flattened_review)

# Save the flattened data to a new JSON file
with open(
    "./kaggle//data/airline_reviews_with_sentiment_analysis.json", "w"
) as outfile:
    json.dump(flattened_data, outfile, indent=4)
