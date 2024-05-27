import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from First.TextProcessing import TextProcessor, process_text

text_processor = TextProcessor()


def process_query(query_text, processor): return process_text(query_text, processor)


corpus_file = r"C:\Users\sayas.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\collection.tsv"
df = pd.read_csv(corpus_file, delimiter='\t', header=None, names=['id', 'text'])

df = df.dropna(subset=['text'])

tfidf_matrix_file = r"C:\Users\sayas.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_matrix.pkl"
with open(tfidf_matrix_file, 'rb') as file: tfidf_matrix = joblib.load(file)

vectorizer_file = r"C:\Users\sayas.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_vectorizer.pkl"
with open(vectorizer_file, 'rb') as file: vectorizer = joblib.load(file)

n = 10

threshold = 0.3


def search_documents(query):
    processed_query = process_query(query, text_processor)
    query_tfidf = vectorizer.transform([processed_query])
    similarity_scores = cosine_similarity(tfidf_matrix, query_tfidf)
    most_relevant_doc_indices = similarity_scores.argsort(axis=0)[-n:].flatten()[::-1]

    retrieved_docs = []
    for i, doc_index in enumerate(most_relevant_doc_indices):
        doc_number = df.iloc[doc_index]['id']
        doc_text = df.iloc[doc_index]['text']

        # Calculate relative similarity
        query_words = query.split()
        doc_words = doc_text.split()
        intersection = len(set(query_words) & set(doc_words))
        relative_similarity = intersection / len(query_words)

        # Check if relative similarity meets the threshold
        if relative_similarity >= threshold:
            retrieved_docs.append(doc_number)

    return retrieved_docs
