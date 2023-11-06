# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:50:22 2023

@author: Danille
"""



import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Your data
x = ["Rock", "Pop", "Jazz", "Hip-Hop", "Classical", "Electronic", "Reggae", "Country", "R&B", "Blues"]
y = ["Energetic", "Energetic", "Relaxing", "Energetic", "Relaxing", "Energetic", "Relaxing", "Energetic", "Energetic", "Relaxing"]

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Combine the features and labels
data = list(zip(x, y_encoded))

# Perform K-means clustering
kmeans = KMeans(n_clusters=2)  # Two clusters for Energetic and Relaxing
kmeans.fit(y_encoded.reshape(-1, 1))

# Get cluster assignments
cluster_labels = kmeans.labels_

# Create lists for Energetic and Relaxing based on the clusters
energetic_genres = [x[i] for i in range(len(x)) if cluster_labels[i] == 0]
relaxing_genres = [x[i] for i in range(len(x)) if cluster_labels[i] == 1]

# Visualize the clusters
plt.figure(figsize=(12, 6))
plt.scatter(y_encoded, [0] * len(y_encoded), c=cluster_labels)

# Label data points with genres
for i, genre in enumerate(x):
    plt.text(y_encoded[i], 0.05, genre, ha='center', fontsize=8)  # Adjusted y-coordinate and font size

# Set custom x-axis limits for better spacing
plt.xlim(-0.5, 1.5)

plt.xlabel("Encoded Mood (0 for Energetic, 1 for Relaxing)")
plt.title("K-means Clustering of Music Genres")
plt.yticks([])
plt.show()

print("Energetic Genres:", energetic_genres)
