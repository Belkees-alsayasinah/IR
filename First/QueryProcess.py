import pickle
import numpy as np
import pandas as pd
import spacy
from annoy import AnnoyIndex

# Load the Transformer language model from spacy
nlp = spacy.load("en_core_web_trf")

# Load the pre-computed word embeddings
with open(r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\spaCy\vectors.pkl", "rb") as f:
    vectors = pickle.load(f)

# Convert the list of word embeddings to a NumPy array
vectors = np.array(vectors)

# Create an Annoy index for approximate nearest neighbors search
index = AnnoyIndex(vectors.shape[1], metric="angular")

# Add the word embeddings to the index
for i, vector in enumerate(vectors):
    index.add_item(i, vector)

# Build the index
index.build(10)  # 10 trees
index.save("index.ann")

# Load the index from disk
index = AnnoyIndex(vectors.shape[1], metric="angular")
index.load("index.ann")

# Load the collection.tsv file into a pandas DataFrame
df = pd.read_csv(r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\collection.tsv", sep='\t', header=None, names=['id', 'text'])

# Define a function to find similar documents
def find_similar_documents(query, n=10):
    # Process the query with the transformer model
    query_vector = nlp(query).vector
    items, distances = index.get_nns_by_vector(query_vector, n, include_distances=True)
    similar_docs = df.iloc[items][['id', 'text']].values
    return similar_docs, distances

# Test the function with a query
query = "how to tell the difference between a girl and boy bearded dragon"
similar_docs, distances = find_similar_documents(query)

print("similar_docs:")
for doc, distance in zip(similar_docs, distances):
    print(f"ID: {doc[0]}, Text: {doc[1]}, Distance: {distance}")
