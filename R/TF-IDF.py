import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load the corpus from TSV
corpus_file = r"C:\Users\sayas\.ir_datasets\antique\result.tsv"
df = pd.read_csv(corpus_file, delimiter='\t', header=None, names=['id', 'text'])

# Drop rows with NaN values in the 'text' column (if any)
df = df.dropna(subset=['text'])
print(f"Number of documents after dropping NaNs: {len(df)}")

# Create a list of documents
documents = df['text'].tolist()
print(f"Number of documents: {len(documents)}")

# R Vectorization
vectorizer = TfidfVectorizer(max_df=0.7,min_df=0.01)

tfidf_matrix = vectorizer.fit_transform(documents)
print(f"Shape of TF-IDF matrix: {tfidf_matrix.shape}")

# Save the vectorizer
vectorizer_file = r"C:\Users\sayas\.ir_datasets\antique\tfidf_vectorizerAA.pkl"
with open(vectorizer_file, 'wb') as file:
    joblib.dump(vectorizer, file)
print(f"TF-IDF vectorizer saved to {vectorizer_file}")

# Save the R matrix
output_file = r"C:\Users\sayas\.ir_datasets\antique\tfidf_matrixAA.pkl"
with open(output_file, 'wb') as file:
    joblib.dump(tfidf_matrix, file)
print(f"TF-IDF matrix saved to {output_file}")
