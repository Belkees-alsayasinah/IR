import pickle

import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, average_precision_score
import numpy as np
import sys
from TextProcessing.TextProcessing import TextProcessor, process_text

sys.path.append('.')


def calculate_precision_recall(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred >= threshold).astype(int)
    precision = precision_score(y_true, y_pred_binary, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred_binary, average='micro', zero_division=0)
    return precision, recall


def calculate_map_score(y_true, y_pred):
    if np.sum(y_true) == 0:
        return 0.0
    return average_precision_score(y_true, y_pred, average='micro')


def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path, delimiter='\t', header=None, names=['pid', 'text'])
    except pd.errors.ParserError as e:
        print(f"Error reading the dataset file: {e}")
        sys.exit(1)
    return data


def load_queries(queries_paths):
    queries = []
    for file_path in queries_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    query = json.loads(line.strip())
                    if 'query' in query:
                        queries.append(query)
                except json.JSONDecodeError:
                    print(f"Skipping invalid line in {file_path}: {line}")
    return queries


def process_texts(texts, processor):
    processed_texts = []
    for text in texts:
        if isinstance(text, str):  # التحقق مما إذا كانت القيمة نصية
            print("text: " + text)
            processed_text = process_text(text, processor)
            processed_texts.append(processed_text)
        else:
            print("Skipping non-string value:", text)
    return processed_texts


def save_tfidf_matrix_and_vectorizer(tfidf_matrix, vectorizer, matrix_file_path, vectorizer_file_path):
    # Save the R matrix
    with open(matrix_file_path, 'wb') as file:
        pickle.dump(tfidf_matrix, file)

    with open(vectorizer_file_path, 'wb') as file:
        pickle.dump(vectorizer, file)


def vectorize_texts(texts):
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError as e:
        print(f"Error during TF-IDF vectorization: {e}")
        print(f"Sample texts: {texts[:5]}")
        sys.exit(1)

    return tfidf_matrix, vectorizer


def get_documents_for_query(query, tfidf_matrix, processor, vectorizer, data):
    processed_query = process_text(query, processor)
    query_vector = vectorizer.transform([processed_query])
    cosine_similarities = cosine_similarity(tfidf_matrix, query_vector).flatten()
    n = 10
    top_documents_indices = cosine_similarities.argsort()[-n:][::-1]
    top_documents = data.iloc[top_documents_indices]
    return top_documents, cosine_similarities[top_documents_indices]


if __name__ == '__main__':
    print("dataset_path")
    processor = TextProcessor()

    dataset_path = r'C:\Users\sayas\.ir_datasets\antique\collection.tsv'
    data = load_dataset(dataset_path)
    data.dropna(subset=['text'], inplace=True)
    data.reset_index(drop=True, inplace=True)  # Reset index here
    print("dataset_path")

    print("Columns in the dataset:", data.columns)
    if 'text' not in data.columns:
        print("The dataset does not contain a 'text' column.")
        sys.exit(1)
    print("start")

    tfidf_matrix, vectorizer = vectorize_texts(data['text'], processor)
    queries_paths = [r'C:\Users\sayas\.ir_datasets\antique\train\relevent_result.jsonl']
    queries = load_queries(queries_paths)

    all_precisions = []
    all_recalls = []
    all_map_scores = []

    for query in queries:
        if 'query' in query:
            processed_query = process_text(query['query'], processor)

            top_documents, cosine_similarities = get_documents_for_query(processed_query, tfidf_matrix, processor,
                                                                         vectorizer, data)

            relevance = np.zeros(len(data))
            for pid in query.get('answer_pids', []):
                relevance[np.where(data['pid'] == pid)[0]] = 1

            y_true = relevance[top_documents.index]

            y_pred = cosine_similarities
            if y_true.sum() == 0:
                continue

            precision, recall = calculate_precision_recall(y_true, y_pred)
            all_precisions.append(precision)
            all_recalls.append(recall)

            map_score = calculate_map_score(y_true, y_pred)
            all_map_scores.append(map_score)

    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_map_score = np.mean(all_map_scores)

    print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average MAP Score: {avg_map_score}")