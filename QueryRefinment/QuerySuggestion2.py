from flask import Flask
from sklearn.feature_extraction.text import TfidfVectorizer
from TextProcessing.TextProcessing import TextProcessor, process_text
import joblib
import json
from sklearn.metrics.pairwise import cosine_similarity
import textdistance
import nltk
from nltk.corpus import wordnet

tfidf_matrix_file = r"C:\Users\sayas\.ir_datasets\antique\Final\tfidf_matrixF.pkl"
with open(tfidf_matrix_file, 'rb') as file:
    tfidf_matrix = joblib.load(file)

vectorizer_file = r"C:\Users\sayas\.ir_datasets\antique\Final\tfidf_vectorF.pkl"
with open(vectorizer_file, 'rb') as file:
    vectorizer = joblib.load(file)
with open(r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\common_words.txt", 'r',
          encoding='utf-8') as file:
    words_to_remove = file.read().splitlines()

processor = TextProcessor()
query_file = r"C:\Users\sayas\.ir_datasets\antique\Final\qas.result.jsonl"
search_file = r"C:\Users\sayas\.ir_datasets\antique\Final\Answers.jsonl"
processed_queries = []
try:
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    query = json.loads(line)["query"]
                    processed_query = process_text(query, processor)
                    processed_queries.append(processed_query)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line}")
                    print(f"Error message: {e}")
            else:
                print("Skipped empty line")
except FileNotFoundError:
    print(f"File not found: {query_file}")

original_queries = []
query_ids = []
try:
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    query = data["query"]
                    query_id = data["qid"]
                    original_queries.append(query)
                    query_ids.append(query_id)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line}")
                    print(f"Error message: {e}")
            else:
                print("Skipped empty line")
except FileNotFoundError:
    print(f"File not found: {query_file}")

search_queries = {}
try:
    with open(search_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    search_queries[data["qid"]] = data["query"]
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line}")
                    print(f"Error message: {e}")
            else:
                print("Skipped empty line")
except FileNotFoundError:
    print(f"File not found: {search_file}")

vectorizer = TfidfVectorizer(max_df=0.85, min_df=2)  # Adjusted parameters
query_vectors = vectorizer.fit_transform(processed_queries)


def refine_query(query):
    query_tokens = nltk.word_tokenize(query)
    refined_query = set(query_tokens)
    for token in query_tokens:
        synsets = wordnet.synsets(token)
        for synset in synsets:
            for lemma in synset.lemmas():
                if lemma.name() != token:
                    refined_query.add(lemma.name())
    return ' '.join(refined_query)


def suggest_similar_queries2(query, n=10, threshold=0.0):
    refined_query = refine_query(query)
    processed_query = process_text(refined_query, processor)
    query_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vector, query_vectors)
    similar_indices = similarities.argsort()[0][-n - 1:-1][::-1]
    similar_queries = [(original_queries[idx], query_ids[idx]) for idx in similar_indices]
    processed_original_query = process_text(query, processor)

    similar_queries = [(q, qid) for q, qid in similar_queries if
                       textdistance.levenshtein.normalized_similarity(processed_original_query,
                                                                      process_text(q, processor)) >= threshold]

    similar_search_queries = [(search_queries[qid], qid) for _, qid in similar_queries if qid in search_queries]

    return similar_search_queries
