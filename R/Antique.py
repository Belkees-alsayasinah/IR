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


# Load words to remove
with open(r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\common_words.txt", 'r',
          encoding='utf-8') as file:
    words_to_remove = file.read().splitlines()


def clean_text(text, words_to_remove):
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in words_to_remove]
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text


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


# Get documents for query function
def get_documents_for_query(query, tfidf_matrix, processor, vectorizer, data):
    processed_query = process_text(query, processor)
    query_vector = vectorizer.transform([processed_query])
    cosine_similarities = cosine_similarity(tfidf_matrix, query_vector).flatten()
    n = 10
    top_documents_indices = cosine_similarities.argsort()[-n:][::-1]
    top_documents = data.iloc[top_documents_indices]
    return top_documents, cosine_similarities[top_documents_indices]


# Main execution block
if __name__ == '__main__':
    processor = TextProcessor()

    dataset_path = r'C:\Users\sayas\.ir_datasets\antique\collection.tsv'
    data = load_dataset(dataset_path)
    data.dropna(subset=['text'], inplace=True)

    if 'text' not in data.columns:
        print("The dataset does not contain a 'text' column.")
        sys.exit(1)

    processed_texts = process_texts(data['text'], processor)

    if not processed_texts:
        print("All documents are empty after preprocessing.")
        sys.exit(1)
        # Paths to save/load the R matrix and vectorizer
    tfidf_matrix_file_path = r'C:\Users\sayas\.ir_datasets\antique\test\tfidf_matrixA.pkl'
    vectorizer_file_path = r'C:\Users\sayas\.ir_datasets\antique\test\tfidf_vectorizerA.pkl'
    tfidf_matrix, vectorizer = vectorize_texts(processed_texts)
    save_tfidf_matrix_and_vectorizer(tfidf_matrix, vectorizer, tfidf_matrix_file_path, vectorizer_file_path)

    # tfidf_matrix, vectorizer = vectorize_texts(data['text'], processor)

    queries_paths = [r'C:\Users\sayas\.ir_datasets\antique\relevent_result.jsonl']
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
            relevant_docs_found = False  # Flag to check if relevant documents were found for this query
            for pid in query.get('answer_pids', []):
                if pid in data['pid'].values:
                    relevant_docs_found = True
                    relevance[np.where(data['pid'] == pid)[0]] = 1

            if relevant_docs_found:
                # Filter out invalid indices that are out of range
                valid_indices = top_documents.index[top_documents.index < len(data)]

                if valid_indices.size > 0:
                    y_true = relevance[np.asarray(valid_indices).astype(int)]
                    y_pred = cosine_similarities[:valid_indices.size]  # Ensure y_pred has the same size as y_true

                    if np.sum(y_true) > 0:  # Check if there are relevant documents in the retrieved results
                        precision, recall = calculate_precision_recall(y_true, y_pred)
                        map_score = calculate_map_score(y_true, y_pred)
                    else:
                        precision, recall, map_score = 0.0, 0.0, 0.0

                    all_precisions.append(precision)
                    all_recalls.append(recall)
                    all_map_scores.append(map_score)

                    print(
                        f"Query ID: {query.get('qid', 'N/A')}, Precision: {precision}, Recall: {recall}, MAP Score: {map_score}")
                    print(f"Top Documents: {top_documents['pid'].tolist()}")
                    print(f"Relevance: {y_true.tolist()}")
                    print(f"Cosine Similarities: {y_pred.tolist()}")
                else:
                    print(f"No relevant documents found within the valid range for query ID: {query.get('qid', 'N/A')}")
            else:
                print(f"No relevant documents found for query ID: {query.get('qid', 'N/A')}")

    # Calculate average performance metrics only if there were relevant documents found
    if all_precisions and all_recalls and all_map_scores:
        avg_precision = np.nanmean(all_precisions)
        avg_recall = np.nanmean(all_recalls)
        avg_map_score = np.nanmean(all_map_scores)

        print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average MAP Score: {avg_map_score}")
    else:
        print("No relevant documents found for any query.")
