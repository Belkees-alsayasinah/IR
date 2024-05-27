import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textdistance
from First.TextProcessing import process_text, TextProcessor

# Load the preprocessor
text_processor = TextProcessor()
# Load the queries file
query_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\qas.result.jsonl"
search_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\qas.search.jsonl"
# Process the queries and store them in a list
processed_queries = []
with open(query_file, 'r', encoding='utf-8') as f:
    for line in f:
        query = json.loads(line)["query"]
        processed_query = process_text(query, text_processor)
        processed_queries.append(processed_query)

# Load the original queries from the file
original_queries = []
query_ids = []
with open(query_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        query = data["query"]
        query_id = data["qid"]
        original_queries.append(query)
        query_ids.append(query_id)

# Load the search queries from the search file into a dictionary
search_queries = {}
with open(search_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        search_queries[data["qid"]] = data["query"]
# Convert processed queries to TF-IDF vectors
vectorizer = TfidfVectorizer(max_df=0.85, min_df=2)  # Adjusted parameters
query_vectors = vectorizer.fit_transform(processed_queries)


# Function to suggest similar queries
def suggest_similar_queries(query, n=10, threshold=0.0):  # Increased n and threshold
    # Process the input query
    processed_query = process_text(query, text_processor)
    # Convert processed query to TF-IDF vector
    query_vector = vectorizer.transform([processed_query])

    # Compute cosine similarity between the input query and each refined query
    similarities = cosine_similarity(query_vector, query_vectors)

    # Find the n most similar queries to the input query
    similar_indices = similarities.argsort()[0][-n - 1:-1][::-1]
    similar_queries = [(original_queries[idx], query_ids[idx]) for idx in similar_indices]

    # Process original query for comparison
    processed_original_query = process_text(query, text_processor)

    # Filter similar queries based on Levenshtein similarity
    similar_queries = [(q, qid) for q, qid in similar_queries if
                       textdistance.levenshtein.normalized_similarity(processed_original_query,
                                                                      process_text(q, text_processor)) >= threshold]

    # Retrieve the corresponding search queries using the qid
    similar_search_queries = [(search_queries[qid], qid) for _, qid in similar_queries if qid in search_queries]

    # Return the list of similar queries and their search counterparts
    return similar_search_queries


# Example usage
query = "do cycling gloves make a sad"
similar_queries = suggest_similar_queries(query, n=10)  # Adjusted n value
print("Query Similar: ======>")
print("Query is: ", query)
for i, (s_query, qid) in enumerate(similar_queries):
    print(f"{i}, Query ID: {qid}, Query: {s_query}")
