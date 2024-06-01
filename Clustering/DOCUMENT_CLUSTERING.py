from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import numpy as np
import joblib
from scipy.sparse import csr_matrix
from umap import UMAP
from sklearn.metrics import silhouette_score, davies_bouldin_score

vectorizer_file = r"C:\Users\sayas\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\tfidf_vectorizerA.pkl"
vectorizer = joblib.load(open(vectorizer_file, 'rb'))
print(f"Loaded TF-IDF vectorizer")

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

        indices = np.where(labels == label)[0]

        tfidf_sum = np.sum(tfidf_matrix[indices], axis=0)

        top_indices = np.argsort(tfidf_sum)[::-1]

        for ind in top_indices[:10]:
            print(f' {terms[ind]}')
        print()


def predict_cluster(vectorizerX, model, svd_model, query):
    Y = vectorizerX.transform(query)

    reduced_query = svd_model.transform(Y)

    distances = model.predict(reduced_query)
    return distances


def plot_clusters(reduced_data, labels):
    plt.figure(figsize=(10, 8))

    unique_labels = set(labels)
    palette = plt.cm.get_cmap('viridis', len(unique_labels))

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


tfidf_matrix = csr_matrix(tfidf_matrix)

umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1)

reduced_tfidf = umap.fit_transform(tfidf_matrix)

model = create_clusters(reduced_tfidf)

top_term_per_cluster(model, vectorizer)

query = ["example query to classify"]
prediction = predict_cluster(vectorizer, model, umap, query)
print(f'The query belongs to cluster {prediction[0]}')

labels = model.labels_
silhouette_avg = silhouette_score(reduced_tfidf, labels)
davies_bouldin = davies_bouldin_score(reduced_tfidf, labels)

print(f"Silhouette Score: {silhouette_avg}")
print(f"Davies-Bouldin Index: {davies_bouldin}")

plot_clusters(reduced_tfidf, labels)
