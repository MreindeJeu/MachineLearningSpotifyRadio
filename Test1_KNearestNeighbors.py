# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:46:06 2023

@author: Danille
"""
#Bron:https://realpython.com/knn-python/ bewerkt met eigen inzicht en ChatGPT

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

# Initial data points
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

# Generate 1000 random x- and y-values and add them to the lists
for _ in range(1000):
    x.append(random.uniform(0, 2000))
    y.append(random.uniform(0, 2500))

data = list(zip(x, y))

# Convert data to a NumPy array for easier manipulation
data = np.array(data)

# Initialize K-Nearest Neighbors with k=
k = 30
nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)

# Generate a random example point within a specified range
random_example_point = np.array([[random.uniform(0, 2000), random.uniform(0, 2500)]])

# Find the k-nearest neighbors for the random example point
distances, indices = nbrs.kneighbors(random_example_point)

# Extract the nearest neighbor IDs and their corresponding coordinates
nearest_neighbor_ids = indices[0]
nearest_neighbor_coordinates = data[nearest_neighbor_ids]

# Calculate the mean of the nearest neighbor 'y' values
nearest_neighbor_rings = nearest_neighbor_coordinates[:, 1]
prediction = nearest_neighbor_rings.mean()

# Plot the data points and the random example point along with its k-nearest neighbors
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], label='Data Points')
plt.scatter(random_example_point[0, 0], random_example_point[0, 1], color='red', marker='x', s=100, label='Random Example Point')
plt.scatter(nearest_neighbor_coordinates[:, 0], nearest_neighbor_coordinates[:, 1], color='green', marker='o', label='Nearest Neighbors')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('K-Nearest Neighbors Visualization')
plt.show()

# Print the mean prediction
print("Predicted Ring Value:", prediction)
