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

# # Print the tagged data to verify
# print(tagged_data)

# Define the model
model = Doc2Vec(vector_size=500, window=2, workers=4, min_count=2, epochs=80)

# Build the vocabulary
model.build_vocab(tagged_data)

# Train the model
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Save the model
model.save("d2v.model")

