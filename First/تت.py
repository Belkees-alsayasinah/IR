import os
import csv
from gensim00.models.doc2vec import Doc2Vec, TaggedDocument

# Load the saved model
model_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\spaCy\Doc2Vec.model"
model = Doc2Vec.load(model_file)

# Load the dataset file
data_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\collection.tsv"

processed_data = []
try:
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) > 1:  # Ensure there are at least 2 columns
                processed_data.append(row[1])
except FileNotFoundError:
    print(f"File not found: {data_file}")
    processed_data = []

# Convert texts to TaggedDocument
tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(processed_data)]

def process_query(query):
    # Infer a vector for the query
    query_vector = model.infer_vector(query.split())

    # Calculate cosine similarity with all documents in the dataset
    sims = model.dv.most_similar([query_vector], topn=5)

    # Return the similar documents
    return sims

# Define a query
query = "rabbit"

# Infer a vector for the query
query_vector = model.infer_vector(query.split())

# Find the most similar documents in the training set
sims = model.dv.most_similar([query_vector], topn=5)

# Print the similarities along with the document numbers
print(f"Query similarities: {[(doc_tag, similarity) for doc_tag, similarity in sims]}")

# Find similar documents in the dataset file
def find_similar_documents(model, query_vector, tagged_data):
    similarities = []
    for doc_tag, doc in tagged_data:
        doc_vector = model.infer_vector(doc)
        similarity = model.cosine_similarities([query_vector], [doc_vector])[0][0]
        similarities.append((doc_tag, similarity))
    return similarities

sims = find_similar_documents(model, query_vector, tagged_data)

# Print the top 5 similar documents
print(f"Top 5 similar documents: {sims[:5]}")



# # Test the function
# query = input("Enter a query: ")
# similar_documents = process_query(query)
# print(f"Similar documents for query '{query}':")
# for sim in similar_documents:
#     print(f"Document {sim[0]} with similarity {sim[1]:.4f}")