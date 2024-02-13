import qdrant_client
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import json
import numpy as np
from embed_reviews_bge_large_en import generate_embedding_bge_large_en
import dotenv
import os
import json
import requests
from datetime import datetime


# from embed_reviews_e5_mistral_7b_instruct import generate_embedding_e5_mistral

url = "http://localhost:32768"
qdrantclient = qdrant_client.QdrantClient(url=url, prefer_grpc=False)
qdrant_collection_name = "airline_industry_customer_review_analysis"
qdrant_collections = qdrantclient.get_collections()

# only create collection if it doesn't exist
if qdrant_collection_name not in [c.name for c in qdrant_collections.collections]:
    qdrantclient.recreate_collection(
        collection_name=qdrant_collection_name,
        vectors_config=models.VectorParams(
            size=1024,
            distance=models.Distance.COSINE,
        ),
    )

base_path = "../../eda/kaggle/data"
input_filename = "airline_reviews_with_sentiment_analysis.json"


# Load JSON data
with open(f"{base_path}/{input_filename}", "r") as file:
    data = json.load(file)

for entry in data:
    # Extract user review
    review_data = entry["Review"]

    # Create embeddings for the review data
    user_review_embedding = generate_embedding_bge_large_en(review_data)
    unique_key = datetime.now().strftime("%Y%m%d%H%M%S%f")

    # Prepare the data entry for qdrant
    data_entry = {
        "id": str(unique_key),
        "values": user_review_embedding.tolist(),
        "metadata": {
            "Airline": entry["Airline"],
            "Review": entry["Review"],
            "Aircraft": entry["Aircraft"],
            "travel_class": entry["Type Of Traveller"],
            "Seat_Type": entry["Seat Type"],
            "Seat_Comfort_Rating": entry["Seat Comfort"],
            "Cabin_Staff_Service_Rating": entry["Cabin Staff Service"],
            "Food_Beverages_Rating": entry["Food & Beverages"],
            "Ground_Service_Rating": entry["Ground Service"],
            "Inflight_Entertainment_Rating": entry["Inflight Entertainment"],
            "Wifi_Connectivity_Rating": entry["Wifi & Connectivity"],
            "Value_For_Money_Rating": entry["Value For Money"],
            "Recommended": entry["Recommended"],
            "Sentiment": entry["Sentiment"],
        },
    }

    operation_info = qdrantclient.upsert(
        collection_name=qdrant_collection_name,
        wait=True,
        points=[
            PointStruct(
                id=data_entry["id"],
                vector=data_entry["values"],
                payload=data_entry["metadata"],
            ),
        ],
    )

    print(operation_info)
