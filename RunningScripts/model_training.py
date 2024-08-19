# combined_script.py

import os
import json
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib

# Paths
hotel_data_path = '../data/game_reviews/'
train_data_path = '../data/processed_train_data.csv'
model_path = '../data/hotel_review_model.pkl'
baseline_path = '../data/new_baseline.txt'

# Step 1: Process hotel reviews
all_reviews = []

for hotel_id in range(1, 1069):  # Now iterating over 1068 hotels
    hotel_file = os.path.join(hotel_data_path, f'{hotel_id}.csv')
    
    if os.path.exists(hotel_file):
        with open(hotel_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                try:
                    # Remove the first two values (assumed to be identifiers)
                    values = row[2:]
                    
                    # Take the last value as the score
                    score = values[-1]
                    
                    # Take the rest as the review features and clean them
                    review_features = ' '.join(values[:-1]).replace(',', '').replace('"', '').replace('\'', '').replace('\n', '').replace(';', '')

                    if ',' in review_features:
                        print(review_features)
                    # Append to the list of all reviews
                    all_reviews.append([review_features, score])
                except:
                    pass

# Save the DataFrame as a new CSV file for training
reviews_df = pd.DataFrame(all_reviews, columns=['review_features', 'hotel_score'])
reviews_df.to_csv(train_data_path, index=False)
print(f"Processed training data saved to {train_data_path}")

# Step 2: Train the model
data = pd.read_csv(train_data_path)
X = data['review_features']
y = data['hotel_score'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

joblib.dump(pipeline, model_path)
print(f"Model saved to {model_path}")

# Step 3: Generate baseline predictions and create baseline file
with open(model_path, 'rb') as model_file:
    model = joblib.load(model_file)

baseline_predictions = {}
for hotel_id in range(1, 1069):
    hotel_file = os.path.join(hotel_data_path, f'{hotel_id}.csv')
    
    if os.path.exists(hotel_file):
        with open(hotel_file, 'r') as file:
            reader = csv.reader(file)
            reviews = []
            for row in reader:
                try:
                    values = row[2:]
                    review_features = ' '.join(values[:-1]).replace(',', '').replace('"', '').replace('\'', '').replace('\n', '').replace(';', '')
                    reviews.append(review_features)
                except:
                    pass

            if reviews:
                review_features = ' '.join(reviews)
                predicted_score = model.predict([review_features])[0]
                baseline_predictions[hotel_id] = predicted_score

with open(baseline_path, 'w') as file:
    json.dump(baseline_predictions, file)
print(f"Baseline predictions saved to {baseline_path}")


