import json

# List of US region airlines to filter
us_specific_airlines = [
    "US Airways",
    "Alaska Airlines",
    "American Airlines",
    "American Eagle",
    "SpiceJet",
    "Virgin America",
    "Delta Air Lines",
    "Southwest Airlines",
    "Spirit Airlines",
    "Frontier Airlines",
    "United Airlines",
    "Hawaiian Airlines",
    "Jetblue Airways",
    "Akasa Air",
]

input_filename = "kaggle/data/airline_reviews_with_sentiment_analysis.json"
output_filename = "kaggle/data/us_airline_reviews_with_sentiment_analysis.json"

# Load the data from the original file
with open(input_filename, "r") as file:
    data = json.load(file)

# Filter the data to only include US based airlines
filtered_data = [entry for entry in data if entry["Airline"] in us_specific_airlines]

# Write the filtered data to a new file
with open(output_filename, "w") as file:
    json.dump(filtered_data, file, indent=4)

print(f"Filtered data written to {output_filename}.")
