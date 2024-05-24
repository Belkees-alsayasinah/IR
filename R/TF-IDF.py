import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Load the corpus from TSV
corpus_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\new_processed_collection.tsv"
df = pd.read_csv(corpus_file, delimiter='\t', header=None, names=['id', 'text'])

# Drop rows with NaN values in the 'text' column (if any)
df = df.dropna(subset=['text'])

# Create a list of documents
documents = df['text'].tolist()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Save the vectorizer
vectorizer_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_vectorizer1.pkl"
with open(vectorizer_file, 'wb') as file:
    joblib.dump(vectorizer, file)

# Save the TF-IDF matrix
output_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_matrix1.pkl"
with open(output_file, 'wb') as file:
    joblib.dump(tfidf_matrix, file)
