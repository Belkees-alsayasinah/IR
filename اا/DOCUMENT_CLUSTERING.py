import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os


def load_documents_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        documents = [line.strip() for line in file if line.strip()]
    return documents


def load_data(file_path):
    df = pd.read_csv(file_path, delimiter='\t', header=None, names=['id', 'text'])
    # Drop rows with NaN values in the 'text' column
    df = df.dropna(subset=['text'])
    documents = df['text'].tolist()
    return df, documents


def apply_kmeans_clustering(tfidf_matrix, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=10)
    labels = kmeans.fit_predict(tfidf_matrix)
    return kmeans, labels


def search(query, documents, labels, tfidf_vectorizer, kmeans, tfidf_matrix, top_n=10):
    print(f"Query: {query}")
    print(f"Length of documents: {len(documents)}")
    print(f"Length of labels: {len(labels)}")

    if len(documents) != len(labels):
        print("Error: The lengths of the documents and labels lists do not match.")
        return []

    query_vector = tfidf_vectorizer.transform([query])
    query_cluster = kmeans.predict(query_vector)[0]
    print(f"The query belongs to cluster: {query_cluster}")

    relevant_docs_indices = [i for i, label in enumerate(labels) if label == query_cluster]
    relevant_docs = [documents[i] for i in relevant_docs_indices]

    if len(relevant_docs) == 0:
        print("No relevant documents found for the given query.")
        return []

    # Calculate cosine similarity between the query and the relevant documents
    relevant_tfidf_matrix = tfidf_matrix[relevant_docs_indices]
    similarities = cosine_similarity(query_vector, relevant_tfidf_matrix).flatten()

    # Get the top_n most similar documents
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_relevant_docs = [relevant_docs[i] for i in top_indices]

    return top_relevant_docs, query_cluster


if __name__ == "__main__":

    # Load data
    file_path = r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\collection.tsv'
    df, documents = load_data(file_path)
    print(f"Number of documents after dropping NaNs: {len(documents)}")

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    print(f"Shape of TF-IDF matrix: {tfidf_matrix.shape}")

    # Save the vectorizer and TF-IDF matrix
    vectorizer_file = r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_vectorizer1.pkl'
    with open(vectorizer_file, 'wb') as file:
        joblib.dump(vectorizer, file)

    tfidf_matrix_file = r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_matrix1.pkl'
    with open(tfidf_matrix_file, 'wb') as file:
        joblib.dump(tfidf_matrix, file)

    # Ensure the number of documents matches the number of rows in the TF-IDF matrix
    if len(documents) != tfidf_matrix.shape[0]:
        print("Error: The number of documents does not match the number of rows in the TF-IDF matrix.")
    else:
        # Apply K-Means clustering
        kmeans, labels = apply_kmeans_clustering(tfidf_matrix)
        print(f"Number of labels: {len(labels)}")
        unique_labels = set(labels)
        print(f"Number of unique clusters: {len(unique_labels)}")
        print(f"Unique clusters: {unique_labels}")

        # Search
        query = input("Enter query: ")
        top_n = 10  # Number of top results to display
        relevant_docs, query_cluster = search(query, documents, labels, vectorizer, kmeans, tfidf_matrix, top_n=top_n)

        # Display relevant documents
        print(f"The query belongs to cluster: {query_cluster}")
        print(f"Top {top_n} Relevant Documents:")
        for doc in relevant_docs:
            print(doc)
