import os
import pandas as pd
import csv

# Path to the hotel review data
hotel_data_path = '../data/game_reviews/'

# List to hold all the processed review data
all_reviews = []

# Iterate over each hotel review file
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
                    
                    # Take the rest as the review features and remove commas if needed
                    review_features = ' '.join(values[:-1]).replace(',', '').replace('"', '').replace('\'', '').replace('\n', '').replace(';', '')

                    if ',' in review_features:
                        print(review_features)
                    # Append to the list of all reviews
                    all_reviews.append([review_features, score])
                except:
                    pass

# Convert the list to a DataFrame
reviews_df = pd.DataFrame(all_reviews, columns=['review_features', 'hotel_score'])

# Save the DataFrame as a new CSV file for training
train_data_path = '../data/processed_train_data.csv'
reviews_df.to_csv(train_data_path, index=False)

print(f"Processed training data saved to {train_data_path}")
