import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

def load_data(file_path):
    df = pd.read_csv(file_path, delimiter='\t', header=None, names=['id', 'text'])
    df = df.dropna(subset=['text'])
    documents = df['text'].tolist()
    return df, documents

def apply_kmeans_clustering(tfidf_matrix, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(tfidf_matrix)
    return kmeans, labels

def search(query, documents, labels, tfidf_vectorizer, kmeans, tfidf_matrix, top_n=10):
    query_vector = tfidf_vectorizer.transform([query])
    query_cluster = kmeans.predict(query_vector)[0]
    relevant_docs_indices = [i for i, label in enumerate(labels) if label == query_cluster]
    relevant_docs = [documents[i] for i in relevant_docs_indices]

    if len(relevant_docs) == 0:
        return []

    relevant_tfidf_matrix = tfidf_matrix[relevant_docs_indices]
    similarities = cosine_similarity(query_vector, relevant_tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_relevant_docs = [relevant_docs[i] for i in top_indices]

    return top_relevant_docs, query_cluster

def plot_clusters(reduced_tfidf, labels, kmeans):
    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    for i, label in enumerate(unique_labels):
        points = reduced_tfidf[labels == label]
        plt.scatter(points[:, 0], points[:, 1], c=[colors(i)], label=f'Cluster {label}')
        plt.scatter(kmeans.cluster_centers_[label, 0], kmeans.cluster_centers_[label, 1], c='black', s=200, alpha=0.5, marker='x')

    plt.title('Cluster Distribution')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    vectorizer_file = r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_vectorizer.pkl'
    with open(vectorizer_file, 'rb') as file:
        vectorizer = joblib.load(file)

    tfidf_matrix_file = r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_matrix.pkl'
    with open(tfidf_matrix_file, 'rb') as file:
        tfidf_matrix = joblib.load(file)

    file_path = r'C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\collection.tsv'
    df, documents = load_data(file_path)

    if len(documents) == tfidf_matrix.shape[0]:
        kmeans, labels = apply_kmeans_clustering(tfidf_matrix)
        unique_labels = set(labels)

        svd = TruncatedSVD(n_components=50)  # زيادة الأبعاد قبل استخدام t-SNE
        reduced_tfidf = svd.fit_transform(tfidf_matrix)
        tsne = TSNE(n_components=2, random_state=0)
        tsne_tfidf = tsne.fit_transform(reduced_tfidf)

        plot_clusters(tsne_tfidf, labels, kmeans)
