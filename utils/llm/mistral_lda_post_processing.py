import requests
import json
import dotenv
import os
import datetime
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from pydantic.json import pydantic_encoder
from data_analysis.us_airline_review_lda import generate_lda_summary_wrapper

dotenv.load_dotenv()

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
model_mistral_tiny = "mistral-tiny"  # Mistral-7B-v0.2
model_mistral_small = "mistral-small"  # Mixtral-8X7B-v0.1
model_mistral_medium = "mistral-medium"  # internal prototype model.
model_mistral_large = "mistral-large-latest" # mistral-large-2407
model = model_mistral_large
client = MistralClient(api_key=MISTRAL_API_KEY)

# input file:
base_path = "../../eda/kaggle/data"
input_filename = "us_airline_reviews_with_sentiment_analysis.json"
input_file = os.path.join(base_path, input_filename)

# output file:
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
base_output_path = "../../eda/kaggle/data/industry_report"
output_filename = f"us_airline_industry_analysis_mistral_{formatted_datetime}.md"
output_file = os.path.join(base_output_path, output_filename)

lda_output_filename = f"us_airline_industry_analysis_lda_mistral_{formatted_datetime}.txt"
lda_output_file = os.path.join(base_output_path, lda_output_filename)

# Get prompt instruction
prompt_augmentation_file = "prompt_instructions_fewshots.json"
with open(prompt_augmentation_file, "r") as file:
    prompt_augmentation = json.load(file)

prompt_instruction = prompt_augmentation["mistral_base_augmentation"]


def mistral_generate_insights_from_cluster_lda(user_query):
    # augmented_query_with_instruction = prompt_instruction + user_query
    # print("augmented_query_with_instruction: ", augmented_query_with_instruction)

    # messages = [ChatMessage(role="user", content=augmented_query_with_instruction)]
    system_message = ChatMessage(role="system", content=prompt_instruction)
    user_message = ChatMessage(role="user", content=user_query)
    messages = [system_message, user_message]

    mistral_response = client.chat(model=model, messages=messages, temperature=0.55)

    # print("complete response ", mistral_response, "\n\n")

    # Convert the response to JSON format
    mistral_response_json = json.dumps(mistral_response, default=pydantic_encoder)

    # Parse the JSON string
    mistral_response_parsed = json.loads(mistral_response_json)

    # Access the message content
    try:
        mistral_response_content = mistral_response_parsed["choices"][0]["message"][
            "content"
        ]
        return mistral_response_content
    except (KeyError, TypeError) as e:
        error_message = f"Error parsing the response: {e}"
        print(error_message)
        return error_message


lda_output = generate_lda_summary_wrapper(input_file)
print("LDA Output: ", lda_output)
generated_insights = mistral_generate_insights_from_cluster_lda(lda_output)
print("Industry Analysis Report: \n", generated_insights)

# Write the Industry Analysis Report to a text file

with open(lda_output_file, 'w') as file:
    file.write("LDA Output: \n" + lda_output + "\n\n")

with open(output_file, 'w') as file:
    file.write("### Industry Analysis Report: \n" + generated_insights)

