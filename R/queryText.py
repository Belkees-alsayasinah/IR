import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from First.TextProcessing import TextProcessor, process_text

text_processor = TextProcessor()

#أول محاولة بحث باستخدام Tf-Idf
def process_query(query_text, processor):
    return process_text(query_text, processor)



corpus_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\collection.tsv"
df = pd.read_csv(corpus_file, delimiter='\t', header=None, names=['id', 'text'])


df = df.dropna(subset=['text'])


tfidf_matrix_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_matrix.pkl"
with open(tfidf_matrix_file, 'rb') as file:
    tfidf_matrix = joblib.load(file)


vectorizer_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_vectorizer.pkl"
with open(vectorizer_file, 'rb') as file:
    vectorizer = joblib.load(file)



def search_documents(query):
    query_tfidf = vectorizer.transform([query])
    similarity_scores = cosine_similarity(tfidf_matrix, query_tfidf)
    most_relevant_doc_indices = similarity_scores.argsort(axis=0)[-n:].flatten()[::-1]
    return df.iloc[most_relevant_doc_indices]



while True:

    query_text = input("Enter your query (or type 'exit' to quit): ")
    if query_text.lower() == 'exit':
        break


    processed_query = process_query(query_text, text_processor)

    query_tfidf = vectorizer.transform([processed_query])


    similarity_scores = cosine_similarity(tfidf_matrix, query_tfidf)

    n = 10

    most_relevant_doc_indices = similarity_scores.argsort(axis=0)[-n:].flatten()[::-1]


    threshold = 0.3


    print("Most relevant documents:")
    for i, doc_index in enumerate(most_relevant_doc_indices):
        doc_number = df.iloc[doc_index]['id']
        doc_text = df.iloc[doc_index]['text']

        query_words = processed_query.split()
        doc_words = doc_text.split()
        intersection = len(set(query_words) & set(doc_words))
        relative_similarity = intersection / len(query_words)

        if relative_similarity >= threshold:
            print(f"Document {doc_number} (Similarity: {similarity_scores[doc_index][0]}):")
            print(doc_text)
            print("\n")

