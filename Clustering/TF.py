import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import scipy.sparse as sp


# Define file paths
corpus_file = r"C:\Users\sayas\.ir_datasets\antique\new_processed_collection.tsv"
vectorizer_file = r"C:\Users\sayas\.ir_datasets\antique\tfidf_vectorizer00.pkl"
tfidf_matrix_file = r"C:\Users\sayas\.ir_datasets\antique\tfidf_matrix00.pkl"

# Load and process the corpus
def load_and_process_data(file_path):
    df = pd.read_csv(file_path, delimiter='\t', header=None, names=['id', 'text'])
    df = df.dropna(subset=['text'])
    documents = df['text'].tolist()
    return df, documents

df, documents = load_and_process_data(corpus_file)
print(f"Number of documents after dropping NaNs: {len(documents)}")

# R Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
print(f"Shape of R matrix: {tfidf_matrix.shape}")

# Save the vectorizer and R matrix
with open(vectorizer_file, 'wb') as file:
    joblib.dump(vectorizer, file)
print(f"R vectorizer saved to {vectorizer_file}")

with open(tfidf_matrix_file, 'wb') as file:
    joblib.dump(tfidf_matrix, file)
print(f"R matrix saved to {tfidf_matrix_file}")


# Load the vectorizer and R matrix
with open(vectorizer_file, 'rb') as file:
    vectorizer = joblib.load(file)

with open(tfidf_matrix_file, 'rb') as file:
    tfidf_matrix = joblib.load(file)

# Ensure the R matrix is in CSR format
if not sp.issparse(tfidf_matrix):
    tfidf_matrix = sp.csr_matrix(tfidf_matrix)

print(f"Shape of R matrix: {tfidf_matrix.shape}")

# Ensure the number of documents matches the R matrix
if len(documents) != tfidf_matrix.shape[0]:
    print("Error: The number of documents does not match the number of rows in the R matrix.")
else:
    print("Data consistency check passed.")
