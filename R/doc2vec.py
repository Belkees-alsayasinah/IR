import csv
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split

# Define the paths
data_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\new_processed_collection.tsv"
model_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\spaCy\Doc2Vec.model"

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

# Ensure there's data to process
if not processed_data:
    raise ValueError("No data found in the file or file is empty.")

# Convert texts to TaggedDocument
tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(processed_data)]

# Split the data into training and validation sets
train_data, val_data = train_test_split(tagged_data, test_size=0.2, random_state=42)

# Train the model
model = Doc2Vec(vector_size=200, window=5, min_count=2, workers=4, epochs=40)
model.build_vocab(train_data)
model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)

# Save the model
os.makedirs(os.path.dirname(model_file), exist_ok=True)
model.save(model_file)


# Define a query
query = "example query"

# Infer a vector for the query
query_vector = model.infer_vector(query.split())

# Find the most similar documents in the training set
sims = model.dv.most_similar([query_vector], topn=5)

# Print the similarities
print(f"Query similarities: {sims}")

# # Evaluate the model
# print("Testing the model...")
# for i in range(5):
#     inferred_vector = model.infer_vector(val_data[i].words)
#     sims = model.dv.most_similar([inferred_vector], topn=5)
#     print(f"Document {val_data[i].tags[0]} similarities: {sims}")
