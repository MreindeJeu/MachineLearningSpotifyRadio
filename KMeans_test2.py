# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:36:19 2023

@author: Danille
"""
# code w3school smet random toegevoegde waarden

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random

# Initial data points
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

# Generate 1000 random x- and y-values and add them to the lists
for _ in range(1000):
    x.append(random.uniform(0, 2000))
    y.append(random.uniform(0, 2500))

data = list(zip(x, y))

# Perform KMeans clustering with 20 clusters
kmeans = KMeans(n_clusters=100)
kmeans.fit(data)

# Generate a list of colors for each cluster
colors = [plt.cm.nipy_spectral(float(i) / kmeans.n_clusters) for i in kmeans.labels_]

# Scatter plot with different colors for each cluster
plt.scatter(x, y, c=colors)
plt.show()

