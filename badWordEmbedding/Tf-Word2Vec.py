import csv
import os
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Define the paths
data_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\new_processed_collection.tsv"
model_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\spaCy\Word2Vec.model"

processed_data = []
try:
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) > 1:  # Ensure there are at least 2 columns
                processed_data.append(row[1].split())  # Split the text into words
except FileNotFoundError:
    print(f"File not found: {data_file}")
    processed_data = []

# Ensure there's data to process
if not processed_data:
    raise ValueError("No data found in the file or file is empty.")

# Join the words to create sentences for TF-IDF
sentences = [' '.join(words) for words in processed_data]

# Split the data into training and validation sets
train_data, val_data, train_sentences, val_sentences = train_test_split(
    processed_data, sentences, test_size=0.2, random_state=42)

# Train the model
model = Word2Vec(sentences=train_data, vector_size=200, window=5, min_count=2, workers=4, epochs=40)

# Train TF-IDF vectorizer
tfidf = TfidfVectorizer()
tfidf.fit(train_sentences)

# Save the model
os.makedirs(os.path.dirname(model_file), exist_ok=True)
model.save(model_file)

# Function to get the weighted average vector for a document using TF-IDF
def get_weighted_average_vector(words, model, tfidf, all_words):
    word_tfidf = tfidf.transform([' '.join(words)])
    vectors = []
    weights = []
    for word in words:
        if word in model.wv:
            tfidf_weight = word_tfidf[0, tfidf.vocabulary_.get(word, 0)]
            vectors.append(model.wv[word] * tfidf_weight)
            weights.append(tfidf_weight)
    if vectors:
        return np.average(vectors, axis=0, weights=weights)
    else:
        return np.zeros(model.vector_size)

# Evaluate the model
print("Testing the model...")
for i in range(5):
    words = val_data[i]
    weighted_avg_vector = get_weighted_average_vector(words, model, tfidf, tfidf.vocabulary_)
    if weighted_avg_vector.any():
        sims = model.wv.similar_by_vector(weighted_avg_vector, topn=5)
        print(f"Words: {words}")
        print(f"Similarities: {sims}")
    else:
        print(f"Words: {words} - No valid vectors found")
