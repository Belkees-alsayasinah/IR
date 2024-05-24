import pandas as pd
import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, average_precision_score
from sklearn.model_selection import train_test_split
import numpy as np
import sys
from First.TextProcessing import TextProcessor, process_text

sys.path.append('.')


# نفسا أفضل نتيجة بس عم احسب الـ Evaluation
def calculate_precision_recall(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred >= threshold).astype(int)
    precision = precision_score(y_true, y_pred_binary, average='micro')
    recall = recall_score(y_true, y_pred_binary, average='micro')
    return precision, recall


def calculate_map_score(y_true, y_pred):
    return average_precision_score(y_true, y_pred, average='micro')


def save_dataset(docs, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for pid, text in enumerate(docs, start=1):
            file.write(f"{pid}\t{text}\n")


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
        print("text: " + text)
        processed_text = process_text(text, processor)
        processed_texts.append(processed_text)
    return processed_texts


def vectorize_texts(data, model):
    vectors = []
    for text in data:
        vector = model.infer_vector(text.split())
        vectors.append(vector)
    return np.array(vectors)


def get_documents_for_query(query, doc_vectors, model, data):
    print("query:" + query)
    query_vector = model.infer_vector(query.split())
    print("doc_vectors:" + doc_vectors)
    print("data:" + query)
    cosine_similarities = cosine_similarity([query_vector], doc_vectors).flatten()
    n = 10
    top_documents_indices = cosine_similarities.argsort()[-n:][::-1]
    print("top_documents_indices:" + top_documents_indices)
    top_documents = data.iloc[top_documents_indices]
    print("top_documents:" + top_documents)
    return top_documents, cosine_similarities[top_documents_indices]


if __name__ == '__main__':
    processor = TextProcessor()

    dataset_path = r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\collection.tsv'
    data = load_dataset(dataset_path)

    print("Columns in the dataset:", data.columns)
    if 'text' not in data.columns:
        print("The dataset does not contain a 'text' column.")
        sys.exit(1)

    processed_texts = process_texts(data['text'], processor)

    # Ensure processed_texts is not empty
    if not processed_texts:
        print("All documents are empty after preprocessing.")
        sys.exit(1)

    tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(processed_texts)]

    train_data, _ = train_test_split(tagged_data, test_size=0.2, random_state=42)

    model = Doc2Vec(vector_size=600, window=5, min_count=2, workers=4, epochs=40)
    model.build_vocab(train_data)
    model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)

    doc_vectors = vectorize_texts(processed_texts, model)

    queries_paths = [r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\qas.forum.jsonl',
                     r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\qas.search.jsonl']
    queries = load_queries(queries_paths)

    all_precisions = []
    all_recalls = []
    all_map_scores = []

    for query in queries:
        if 'query' in query:
            processed_query = process_text(query['query'], processor)
            print("Processed Query:", processed_query)
            top_documents, cosine_similarities = get_documents_for_query(processed_query, doc_vectors, model, data)

            relevance = np.zeros(len(data))
            for pid in query.get('answer_pids', []):
                relevance[np.where(data['pid'] == pid)[0]] = 1

            y_true = relevance[top_documents.index]
            y_pred = cosine_similarities

            if y_true.sum() == 0:
                print(f"No relevant documents for query ID: {query.get('qid', 'N/A')}")
                continue

            precision, recall = calculate_precision_recall(y_true, y_pred)
            all_precisions.append(precision)
            all_recalls.append(recall)

            map_score = calculate_map_score(y_true, y_pred)
            all_map_scores.append(map_score)

            print(
                f"Query ID: {query.get('qid', 'N/A')}, Precision: {precision}, Recall: {recall}, MAP Score: {map_score}")

    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_map_score = np.mean(all_map_scores)

    print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average MAP Score: {avg_map_score}")
