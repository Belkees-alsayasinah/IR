import gensim
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

from First.TextProcessing import TextProcessor, process_text

# Path to the file containing the documents
file_path = "C:/Users/sayas/.ir_datasets/lotte/lotte_extracted/lotte/lifestyle/dev/try.tsv"

# Read the documents from the file
with open(file_path, "r", encoding="utf-8") as file:
    data = file.readlines()

# Tokenize and tag the data
tagged_data = [TaggedDocument(words=word_tokenize(process_text(_d.strip(), TextProcessor()).lower()), tags=[str(i)]) for
               i, _d in enumerate(data)]

# Print the tagged data to verify
print(tagged_data)

# Define the model
model = Doc2Vec(vector_size=300, window=5, workers=4, min_count=2, epochs=80)

# Build the vocabulary
model.build_vocab(tagged_data)

# Train the model
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Save the model
model.save("d2v.model")

# Load the model
model = Doc2Vec.load("d2v.model")


# Function to find the most similar documents to a given text
def find_similar_docs(model, text, topn=4):
    tokens = word_tokenize(process_text(text, TextProcessor()).lower())
    new_vector = model.infer_vector(tokens)
    similar_docs = model.dv.most_similar([new_vector], topn=topn)
    return similar_docs


# Main loop to accept user input from console and find similar documents
if __name__ == "__main__":
    while True:
        user_input = input("Enter a text to find similar documents (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        similar_docs = find_similar_docs(model, user_input)
        print("\nThe most similar documents are:")
        for doc_id, similarity in similar_docs:
            print(f"Document ID: {doc_id}, Similarity: {similarity:.4f}")
            print(f"Content: {data[int(doc_id)]}\n")
