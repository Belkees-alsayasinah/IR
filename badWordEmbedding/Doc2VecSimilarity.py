from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from First.TextProcessing import TextProcessor, process_text
#البحث عن نتائج الاستعلامات - محاولة فاشلة أيضاً
processor = TextProcessor()
class Doc2VecSimilarity:
    def __init__(self, model_path, data_path):
        self.model = Doc2Vec.load(model_path)
        self.data = pd.read_csv(data_path, sep='\t', header=None)

    def vectorize_query(self, query):
        return self.model.infer_vector(query)

    def calculate_similarity(self, query_vector):
        data_vectors = [self.model.dv[i] for i in range(len(self.data))]
        similarities = cosine_similarity([query_vector], data_vectors)[0]
        return similarities

    def search_similar_documents(self, query, top_n=5):
        processed_query = process_text(query, processor)
        query_vector = self.vectorize_query(processed_query.split())
        similarities = self.calculate_similarity(query_vector)
        top_indices = np.argsort(similarities)[::-1][:top_n]
        top_similarities = [similarities[i] for i in top_indices]
        top_documents = [(i, self.get_document_text(i)) for i in top_indices]
        return list(zip(top_documents, top_similarities))

    def get_document_text(self, doc_index):
        return self.data.iloc[doc_index][1]

# Define paths
model_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\doc2vec\Doc2Vec.model"
data_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\collection.tsv"


doc2vec_similarity = Doc2VecSimilarity(model_file, data_file)


query = "what is the difference between a percolator and an espresso make"
results = doc2vec_similarity.search_similar_documents(query)
print("Top similar documents:")
for idx, result in enumerate(results, start=1):
    (doc_number, document_text), similarity = result
    print(f"{idx}. Document {doc_number}:")
    print(document_text)
    print(f"Similarity: {similarity}")
    print()
