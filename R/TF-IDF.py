import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load the corpus from TSV
corpus_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\result.tsv"
df = pd.read_csv(corpus_file, delimiter='\t', header=None, names=['id', 'text'])

# Drop rows with NaN values in the 'text' column (if any)
df = df.dropna(subset=['text'])
print(f"Number of documents after dropping NaNs: {len(df)}")

# Create a list of documents
documents = df['text'].tolist()
print(f"Number of documents: {len(documents)}")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
print(f"Shape of TF-IDF matrix: {tfidf_matrix.shape}")

# Save the vectorizer
vectorizer_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_vectorizer.pkl"
with open(vectorizer_file, 'wb') as file:
    joblib.dump(vectorizer, file)
print(f"TF-IDF vectorizer saved to {vectorizer_file}")

# Save the TF-IDF matrix
output_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_matrix.pkl"
with open(output_file, 'wb') as file:
    joblib.dump(tfidf_matrix, file)
print(f"TF-IDF matrix saved to {output_file}")
