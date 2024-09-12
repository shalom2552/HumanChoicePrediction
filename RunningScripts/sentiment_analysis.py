import os
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# Initialize BERT tokenizer and model from pretrained resources
bert_model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
bert_tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Set the directory containing the CSV files
directory = '../data/game_reviews'

# Function to tokenize and process a review for BERT
def tokenize_review(text):
    return bert_tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)

# Function to classify sentiment from the review text
def classify_sentiment(text):
    tokens = tokenize_review(text)
    with torch.no_grad():
        result = bert_model(**tokens)
    probabilities = torch.softmax(result.logits, dim=1).numpy()[0]
    confidence = np.max(probabilities)
    sentiment_label = np.argmax(probabilities)

    # Categorize sentiment: Merge very negative/negative and positive/very positive
    if sentiment_label in [0, 1]:
        sentiment = 'negative'
    elif sentiment_label in [3, 4]:
        sentiment = 'positive'
    else:
        sentiment = 'negative'  # Consider neutral as negative by default
    return sentiment, confidence

# Function to handle a single CSV file
def analyze_reviews_in_file(file_path):
    data = pd.read_csv(file_path, header=None)
    analysis_results = []
    
    for index, row in data.iterrows():
        review_part1 = str(row[2]) if pd.notna(row[2]) else ""
        review_part2 = str(row[3]) if pd.notna(row[3]) else ""
        full_review = f"{review_part1} {review_part2}".strip().replace('\n', ' ').replace('\r', ' ')
        
        if full_review:  # Only process non-empty reviews
            sentiment, confidence = classify_sentiment(full_review)
            analysis_results.append((row[0], row[1], full_review, sentiment, confidence))
    
    return analysis_results

# Function to process all CSV files in the specified directory
def analyze_all_files(data_dir):
    complete_results = []
    for filename in os.listdir(data_dir):
        print(filename)
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            file_results = analyze_reviews_in_file(file_path)
            complete_results.extend(file_results)
    return complete_results

# Run the analysis and save results to a text file
if __name__ == "__main__":
    overall_results = analyze_all_files(directory)
    
    with open('sentiment_results.txt', 'w', encoding='utf-8') as output_file:
        for review_id, hotel, review_text, sentiment, score in overall_results:
            output_file.write(f"{review_id}\t{hotel}\t{review_text}\t{sentiment}\t{score:.2f}\n")
    
    print("Sentiment analysis complete, results saved to 'sentiment_results.txt'.")
