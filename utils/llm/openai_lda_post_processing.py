from openai import OpenAI
import requests
import json
import dotenv
import os
import re
import datetime
from data_analysis.us_airline_review_lda import generate_lda_summary_wrapper

dotenv.load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY2")
client = OpenAI(api_key=OPENAI_API_KEY)

# input file:
base_path = "../../eda/kaggle/data"
input_filename = "us_airline_reviews_with_sentiment_analysis.json"
input_file = os.path.join(base_path, input_filename)


# output file:
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
base_output_path = "../../eda/kaggle/data/industry_report"
output_filename = f"us_airline_industry_analysis_openai_{formatted_datetime}.md"
output_file = os.path.join(base_output_path, output_filename)
lda_output_filename = f"us_airline_industry_analysis_lda_openai_{formatted_datetime}.txt"
lda_output_file = os.path.join(base_output_path, lda_output_filename)


prompt_augmentation_file = "prompt_instructions_fewshots.json"
with open(prompt_augmentation_file, "r") as file:
    prompt_augmentation = json.load(file)

prompt_instruction = prompt_augmentation["openai_base_augmentation"]


def format_response(api_response):
    formatted_response = api_response.replace("\\n", "\n")
    formatted_response = re.sub(r"###\s*", "", formatted_response)
    formatted_response = re.sub(r"\*\*", "", formatted_response)
    formatted_response = re.sub(r"-\s\*\*", "", formatted_response)
    return formatted_response


def openai_generate_insights_from_cluster_lda(user_query):
    try:
        openai_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": prompt_instruction,
                },
                {"role": "user", "content": user_query},
            ],
            temperature=0.75,
        )

        # print("full response:", openai_response, "\n")

        industry_report = openai_response.choices[0].message.content
        # print("Industry Analysis report before formatting:", industry_report, "\n")
        # industry_report_formatted = format_response(industry_report)

        return industry_report

    except Exception as e:
        print("Error during API call:", e)
        return "An error occurred while generating insights."


lda_output = generate_lda_summary_wrapper(input_file)
print("LDA Output: ", lda_output)
generated_insights = openai_generate_insights_from_cluster_lda(lda_output)
print("Industry Analysis Report:\n", generated_insights)


# Write the Industry Analysis Report to a text file
with open(lda_output_file, 'w') as file:
    file.write("LDA Output: \n" + lda_output + "\n\n")

with open(output_file, 'w') as file:
    file.write("### Industry Analysis Report: \n" + generated_insights)

