import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np
import joblib
from scipy.sparse import csr_matrix
from umap import UMAP
from sklearn.metrics import silhouette_score, davies_bouldin_score

vectorizer_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_vectorizerA.pkl"
vectorizer = joblib.load(open(vectorizer_file, 'rb'))
print(f"Loaded TF-IDF vectorizer")

# Load the TF-IDF matrix
tfidf_matrix_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_matrixA.pkl"
tfidf_matrix = joblib.load(tfidf_matrix_file)
print(f"Loaded TF-IDF matrix with shape: {tfidf_matrix.shape}")


def create_clusters(doc_vector, num_clusters=5):
    model = MiniBatchKMeans(n_clusters=num_clusters, batch_size=2000)
    model.fit(doc_vector)
    return model


def top_term_per_cluster(model, vectorizerX):
    labels = model.labels_
    unique_labels = set(labels)
    terms = vectorizerX.get_feature_names_out()

    for label in unique_labels:
        print(f"Cluster {label}:")

        # Find indices of documents in the current cluster
        indices = np.where(labels == label)[0]

        # Sum the tf-idf values for all documents in the cluster
        tfidf_sum = np.sum(tfidf_matrix[indices], axis=0)

        # Sort terms by tf-idf values
        top_indices = np.argsort(tfidf_sum)[::-1]

        for ind in top_indices[:10]:
            print(f' {terms[ind]}')
        print()  # To separate clusters visually


def predict_cluster(vectorizerX, model, svd_model, query):
    # Transform the query into tf-idf representation
    Y = vectorizerX.transform(query)

    # Reduce the dimensionality of the query using the same SVD model
    reduced_query = svd_model.transform(Y)

    # Predict the cluster
    distances = model.predict(reduced_query)
    return distances


def plot_clusters(reduced_data, labels):
    plt.figure(figsize=(10, 8))

    # Define a color palette
    unique_labels = set(labels)
    palette = plt.cm.get_cmap('viridis', len(unique_labels))

    # Scatter plot for each cluster
    for cluster in unique_labels:
        color = palette(cluster)
        plt.scatter(reduced_data[labels == cluster, 0],
                    reduced_data[labels == cluster, 1],
                    label=f'Cluster {cluster}',
                    s=50,
                    alpha=0.7,
                    edgecolors='k',
                    color=[color])

    plt.title('Clusters Visualization')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()


# Ensure the TF-IDF matrix is in sparse format
tfidf_matrix = csr_matrix(tfidf_matrix)

# Perform dimensionality reduction using UMAP
umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1)

reduced_tfidf = umap.fit_transform(tfidf_matrix)

# Create clusters using the reduced TF-IDF matrix
model = create_clusters(reduced_tfidf)

# Print top terms per cluster
top_term_per_cluster(model, vectorizer)

# Predict cluster for a new queryA
query = ["example query to classify"]
prediction = predict_cluster(vectorizer, model, umap, query)
print(f'The query belongs to cluster {prediction[0]}')

# Evaluate clustering quality
labels = model.labels_
silhouette_avg = silhouette_score(reduced_tfidf, labels)
davies_bouldin = davies_bouldin_score(reduced_tfidf, labels)

print(f"Silhouette Score: {silhouette_avg}")
print(f"Davies-Bouldin Index: {davies_bouldin}")

# Plot the clusters
plot_clusters(reduced_tfidf, labels)

