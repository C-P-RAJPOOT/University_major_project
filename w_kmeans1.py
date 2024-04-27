import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
    
    def save_plot(self, X, labels, file_name):
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='*', s=300, c='red', label='Centroids')
        plt.legend()
        plt.title('KMeans Clustering')
        plt.xlabel('math score')
        plt.ylabel('reading score')
        plt.savefig(file_name)
        #plt.show()
        plt.close()


def run_kmeans():
    # Load data from a CSV file
    csv_file = 'StudentsPerformance.csv'
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Check if 'math score' and 'reading score' columns exist in the DataFrame
    if 'math score' not in df.columns or 'reading score' not in df.columns:
        print("Required columns 'math score' and 'reading score' not found in the DataFrame.")
        return

    # Select relevant features for clustering and drop rows with missing values
    X = df[['math score', 'reading score']].dropna().values

    # Fit KMeans model
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)

    # Get cluster labels for the data
    labels = kmeans.predict(X)

    # Plot clusters
    kmeans.save_plot(X, labels, 'kmeans_plot.svg')
    def read_svg_file(file_path):
        try:
            with open(file_path, 'r') as file:
                svg_data = file.read()
            return svg_data
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None

# Example usage:
    file_path = 'kmeans_plot.svg'
    svg_data = read_svg_file(file_path)
    #if svg_data:
        #print("SVG file contents:")
        #print(svg_data)


    return svg_data

#run_kmeans()
