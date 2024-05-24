import csv
import os
from gensim00.models import Word2Vec
from sklearn.model_selection import train_test_split

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

# Split the data into training and validation sets
train_data, val_data = train_test_split(processed_data, test_size=0.2, random_state=42)

# Train the model
model = Word2Vec(sentences=train_data, vector_size=200, window=5, min_count=2, workers=4, epochs=40)

# Save the model
os.makedirs(os.path.dirname(model_file), exist_ok=True)
model.save(model_file)

# Evaluate the model
print("Testing the model...")
for i in range(5):
    words = val_data[i]
    inferred_vector = model.wv[words]  # Get the vectors for each word
    if len(inferred_vector) > 0:
        avg_vector = inferred_vector.mean(axis=0)  # Average the vectors
        sims = model.wv.similar_by_vector(avg_vector, topn=5)
        print(f"Words: {words}")
        print(f"Similarities: {sims}")
    else:
        print(f"Words: {words} - No valid vectors found")
