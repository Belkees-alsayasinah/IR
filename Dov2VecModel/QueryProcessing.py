import gensim
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from First.TextProcessing import TextProcessor, process_text

class Doc2VecModel:
    def __init__(self, model_path):
        self.model = Doc2Vec.load(model_path)

    def find_similar_docs(self, text, topn=10):
        tokens = word_tokenize(process_text(text, TextProcessor()).lower())
        new_vector = self.model.infer_vector(tokens)
        similar_docs = self.model.dv.most_similar([new_vector], topn=topn)
        return similar_docs

if __name__ == "__main__":
    # Path to the file containing the documents
    file_path = "C:/Users/sayas/.ir_datasets/lotte/lotte_extracted/lotte/lifestyle/dev/try.tsv"

    # Read the documents from the file
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.readlines()


    model_path = "d2v.model"
    doc2vec_model = Doc2VecModel(model_path)

    while True:
        user_input = input("Enter a text to find similar documents (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        similar_docs = doc2vec_model.find_similar_docs(user_input)
        print("\nThe most similar documents are:")
        for doc_id, similarity in similar_docs:
            print(f"Document ID: {doc_id}, Similarity: {similarity:.4f}")
            print(f"Content: {data[int(doc_id)]}\n")
