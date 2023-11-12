import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist
from pyclustering.cluster.kmedoids import kmedoids

 # Creeer dataset
data, _ = make_blobs(n_samples=300, centers=4, random_state=42)
df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])

 # Visualiseer data zonder clustering algoritme 
plt.scatter(df['Feature1'], df['Feature2'])
plt.title('Generated Data')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(df)

 # Visualiseer clusters K-means
plt.scatter(df['Feature1'], df['Feature2'], c=df['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.show()



def kmedians(X, k, max_iters=100):
    # Initialiseer medians
    medians = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iters):
        # Verbind elke blob met dichtsbijzijnde median
        labels = np.argmin(cdist(X, medians, metric='euclidean'), axis=1)

        # Update median
        new_medians = np.array([np.median(X[labels == j], axis=0) for j in range(k)])

        # Check overeenkomst
        if np.all(medians == new_medians):
            break

        medians = new_medians

    return labels, medians

# K-median
labels, medians = kmedians(df[['Feature1', 'Feature2']].values, k=4)

# Visualisatie van K-median clustering
plt.scatter(df['Feature1'], df['Feature2'], c=labels, cmap='viridis')
plt.scatter(medians[:, 0], medians[:, 1], s=300, c='red', marker='X', label='Medians')
plt.title('K-medians Clustering')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.show()



# K-medoid
initial_medoids = np.random.choice(df.index, size=4, replace=False)
kmedoids_instance = kmedoids(df[['Feature1', 'Feature2']].values, initial_medoids)
kmedoids_instance.process()

# Clustering van Medoids
clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()

# Visualisatie van clusters
colors = np.zeros(len(df))
for i, cluster in enumerate(clusters):
    colors[cluster] = i

plt.scatter(df['Feature1'], df['Feature2'], c=colors, cmap='viridis')
plt.scatter(df.iloc[medoids]['Feature1'], df.iloc[medoids]['Feature2'], s=300, c='red', marker='X', label='Medoids')
plt.title('K-medoids Clustering')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.show()
