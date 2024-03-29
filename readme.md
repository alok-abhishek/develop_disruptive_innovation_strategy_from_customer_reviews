# Detailed description
- **Link to blog**: [Disrupting Industries with Data Science: Machine Learning as the Key to Disruptive Innovation](https://alokabhishek.com/product-management/disrupting-industries-with-data-science-machine-learning-as-the-key-to-disruptive-innovation/)
- Develop actionable innovation strategies by harnessing the power of advanced machine learning techniques such as RoBERTa, BART, LLM, & LDA

# Customer Reviews to Industry Analysis Report with innovation/disruption strategy generator Tool
This repository houses a Python tool designed to process user reviews and product industry analysis report covering topics such as:
1. Do sentiment analysis and create clusters of user reviews based on sentiment. 
2. Perform LDA analysis for each of these clusters and summarize the overall findings from the LDA analysis. 
3. Conduct a SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis based on the LDA findings. 
4. Identify opportunities for market disruption in the airline industry. 
5. Provide insights on applying Blue Ocean Strategy to create new market space. 
6. Suggest hypothesis tests with description of null hypothesis (H0) and alternative hypothesis (Ha or H1) to validate these findings.  

*This tool is a prototype to show proof of concept intended for education and informational purposes only. Generative AI and LLM can hallucinate and make errors therefore users are advised to make business decision only after human review and involvement.* 

## Features

- **Sentiment Analysis**: Reads PDF files, with the option to exclude specific pages as provided by the user (references and some other pages towards the end of the report).
- **Industry Report Generation**: A report containing strategic innovation ideas for industry is generated and saved in eda/kaggle/data/industry_report. The tool uses OpenAI's GPT 4 model and Mistral model both of which require a paid API key. But code can be modified to work with other open source models.

## How to use - Input and output file set up
- **Setting up input file**: default input csv file path/directory is ../../eda/kaggle/data. The code creates a json file from csv in the same director and perform all the operations on json file.  
- **Setting up output file**: default output folder where text file with industry report is created is ../../eda/kaggle/data/industry_report
- **Setting up Kaggle, Mistral, and OpenAI API**: Rename the .env-example file to .env file and update it with your OpenAI, Mistral AI and Kaggle API key.
- **Install required libraries**: use pip install -r requirements.txt to install required python packages to run the program.
- **How to run**: After setting up API keys, execute openai_lda_post_processing.py and mistral_lda_post_processing.py to run the tool for OpenAI and Mistral LLM respectively.

