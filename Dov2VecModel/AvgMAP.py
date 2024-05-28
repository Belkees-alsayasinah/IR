import pandas as pd
import json
from sklearn.metrics import precision_score, recall_score, average_precision_score
import numpy as np
import sys
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from textblob import TextBlob

from TextProcessing.TextProcessing import TextProcessor, process_text
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, \
    confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('.')


def calculate_precision_recall(y_true, y_pred, threshold=0.0):
    y_pred_binary = (np.array(y_pred) >= threshold).astype(int)
    precision = precision_score(y_true, y_pred_binary, average='micro')
    recall = recall_score(y_true, y_pred_binary, average='micro')
    return precision, recall


def calculate_map_score(y_true, y_pred):
    return average_precision_score(y_true, y_pred, average='micro')


# Save dataset function
def save_dataset(docs, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for pid, text in enumerate(docs, start=1):
            file.write(f"{pid}\t{text}\n")


# Load dataset function
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path, delimiter='\t', header=None, names=['pid', 'text'])
    except pd.errors.ParserError as e:
        print(f"Error reading the dataset file: {e}")
        sys.exit(1)
    return data


# Load queries function
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


# Clean text function
def clean_text(text, words_to_remove):
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in words_to_remove]
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text


# Process texts function
def process_texts(texts, processor):
    processed_texts = []
    for text in texts:
        processed_text = process_text(text, processor)
        processed_texts.append(processed_text)
    return processed_texts


# Train Doc2Vec model function
def train_doc2vec_model(data, processor):
    tagged_data = [
        TaggedDocument(words=word_tokenize(process_text(row['text'], processor).lower()), tags=[str(row['pid'])]) for
        index, row in data.iterrows()]
    model = Doc2Vec(vector_size=300, window=5, workers=4, min_count=20, epochs=80)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def context_aware_spelling_correction(self, text):
        return str(TextBlob(text).correct())
# Get documents for query function using Doc2Vec
def get_documents_for_query(query, model, processor, data):
    processed_query = process_text(query, processor)
    query_vector = model.infer_vector(word_tokenize(processed_query.lower()))
    similar_docs = model.dv.most_similar([query_vector], topn=10)
    top_documents_indices = [int(doc_id) for doc_id, sim in similar_docs]
    top_documents = data[data['pid'].isin(top_documents_indices)]
    cosine_similarities = [sim for doc_id, sim in similar_docs]
    return top_documents, cosine_similarities


# Main execution block
if __name__ == '__main__':
    processor = TextProcessor()

    dataset_path = r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\collection.tsv'
    data = load_dataset(dataset_path)

    if 'text' not in data.columns:
        print("The dataset does not contain a 'text' column.")
        sys.exit(1)

    processed_texts = process_texts(data['text'], processor)
    if not processed_texts:
        print("All documents are empty after preprocessing.")
        sys.exit(1)

    model = train_doc2vec_model(data, processor)
    model.save("d2v.model")
    model = Doc2Vec.load("d2v.model")

    queries_paths = [
        r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\qas.search.jsonl']
    queries = load_queries(queries_paths)

    all_precisions = []
    all_recalls = []
    all_map_scores = []

    for query in queries:
        if 'query' in query:
            top_documents, cosine_similarities = get_documents_for_query(query['query'], model, processor, data)
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
