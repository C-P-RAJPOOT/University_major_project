import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        n_samples, n_features = X.shape
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            labels = self._assign_labels(X)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def _assign_labels(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

def run_kmeans_clustering(csv_file='StudentsPerformance.csv', n_clusters=4, max_iter=300, save_path='kmeans_image.png'):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

    if 'math score' not in df.columns or 'reading score' not in df.columns:
        print("Required columns 'math score' and 'reading score' not found in the DataFrame.")
        return None

    X = df[['math score','reading score']].dropna().values

    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter)
    kmeans.fit(X)

    labels = kmeans.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], marker='*', s=300, c='red', label='Centroids')
    plt.legend()
    plt.title('KMeans Clustering')
    plt.xlabel('math score')
    plt.ylabel('reading score')
    plt.savefig(save_path)  # Save the plot as an image
    plt.close()  # Close the plot to free up memory

    return save_path  # Return the path to the saved image

# Example usage:
run_kmeans_clustering(save_path='kmeans_image.png')
