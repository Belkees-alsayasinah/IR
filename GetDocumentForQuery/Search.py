import sys
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from TextProcessing.TextProcessing import process_text, TextProcessor


def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path, delimiter='\t', header=None, names=['pid', 'text'])
    except pd.errors.ParserError as e:
        print(f"Error reading the dataset file: {e}")
        sys.exit(1)
    return data


def get_documents_for_query_dataset1(query):
    dataset1_path = r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\collection.tsv'
    tfidf_matrix1_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_matrix.pkl"
    vectorizer1_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_vectorizer.pkl"

    data1 = load_dataset(dataset1_path)
    data1.dropna(subset=['text'], inplace=True)

    with open(tfidf_matrix1_file, 'rb') as file:
        tfidf_matrix1 = joblib.load(file)

    with open(vectorizer1_file, 'rb') as file:
        vectorizer1 = joblib.load(file)

    processor = TextProcessor()
    processed_query = process_text(query, processor)
    query_vector = vectorizer1.transform([processed_query])
    cosine_similarities = cosine_similarity(tfidf_matrix1, query_vector).flatten()

    n = 10
    top_documents_indices = cosine_similarities.argsort()[-n:][::-1]
    top_documents = data1.iloc[top_documents_indices]
    return top_documents, cosine_similarities[top_documents_indices]


def get_documents_for_query_dataset2(query):
    dataset2_path = r'C:\Users\sayas\.ir_datasets\antique\collection.tsv'
    tfidf_matrix2_file = r"C:\Users\sayas\.ir_datasets\antique\tfidf_matrix.pkl"
    vectorizer2_file = r"C:\Users\sayas\.ir_datasets\antique\tfidf_vectorizer.pkl"

    data2 = load_dataset(dataset2_path)
    data2.dropna(subset=['text'], inplace=True)

    with open(tfidf_matrix2_file, 'rb') as file:
        tfidf_matrix2 = joblib.load(file)

    with open(vectorizer2_file, 'rb') as file:
        vectorizer2 = joblib.load(file)

    processor = TextProcessor()
    processed_query = process_text(query, processor)
    query_vector = vectorizer2.transform([processed_query])
    cosine_similarities = cosine_similarity(tfidf_matrix2, query_vector).flatten()

    n = 10
    top_documents_indices = cosine_similarities.argsort()[-n:][::-1]
    # print("Top document indices:", top_documents_indices)

    # Check if any of the indices exceed the length of data2
    if any(idx >= len(data2) for idx in top_documents_indices):
        print("Error: Top document indices out-of-bounds")
        # Handle the error appropriately, such as returning an empty DataFrame
        return pd.DataFrame(), []

    top_documents = data2.iloc[top_documents_indices]
    return top_documents, cosine_similarities[top_documents_indices]
