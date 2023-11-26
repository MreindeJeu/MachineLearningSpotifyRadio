# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:27:42 2023

@author: Danille
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
import pandas as pd

# Load data from the CSV file
file_path = r"C:\Users\Danille\Documents\minor\Block2\Tech\Test_KNearestNeighbors\Data\features_30_sec.csv"
df = pd.read_csv(file_path)

# Extract the desired columns for x, y, and labels
x_column = "chroma_stft_mean"
y_column = "chroma_stft_var"
label_column = "label"
data = df[[x_column, y_column, label_column]].values

# Convert data to numeric type
data[:, :2] = data[:, :2].astype(float)

# Initialize K-Nearest Neighbors with k=
k = 60
nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data[:, :2])

# Generate a random example point within a specified range
random_example_point = np.array([random.uniform(data[:, 0].min(), data[:, 0].max()),
                                 random.uniform(data[:, 1].min(), data[:, 1].max())]).reshape(1, -1)

# Find the k-nearest neighbors for the random example point
distances, indices = nbrs.kneighbors(random_example_point)

# Extract the nearest neighbor IDs and their corresponding coordinates
nearest_neighbor_ids = indices[0]
nearest_neighbor_coordinates = data[nearest_neighbor_ids]

# Calculate the mean of the nearest neighbor 'y' values
nearest_neighbor_rings = nearest_neighbor_coordinates[:, 1]
prediction = nearest_neighbor_rings.mean()

# Plot the data points with different colors based on labels
labels = np.unique(data[:, 2])
colors = plt.cm.rainbow(np.linspace(0, 1, len(labels)))
for i, label in enumerate(labels):
    label_mask = data[:, 2] == label
    plt.scatter(data[label_mask, 0], data[label_mask, 1], label=f'Label {label}', c=[colors[i]], alpha=0.6, marker='.')

# Plot the random example point
plt.scatter(random_example_point[0, 0], random_example_point[0, 1], color='red', marker='x', s=100, label='Random Example Point')

# Plot the nearest neighbors with their original colors based on labels
for neighbor_id in nearest_neighbor_ids:
    label_color = colors[np.where(labels == data[neighbor_id, 2])[0][0]]
    plt.scatter(data[neighbor_id, 0], data[neighbor_id, 1], color=label_color, marker='.', s=100)

plt.xlabel('X')
plt.ylabel('Y')

# Move legend outside the plot area
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title('K-Nearest Neighbors Visualization with chroma_stft_ -mean and -var')
plt.show()

# Print the filenames of the nearest neighbors
nearest_neighbor_filenames = df.iloc[nearest_neighbor_ids]["filename"]
print("Filenames of Nearest Neighbors:")
print(nearest_neighbor_filenames)

# Add priority points to a new Excel file
priority_file_path = r"C:\Users\Danille\Documents\minor\Block2\Tech\Test_KNearestNeighbors\Data\DOC1_priotiteit.xlsx"

# Read the existing priority file if it exists, or create a new DataFrame
try:
    priority_df = pd.read_excel(priority_file_path)
except FileNotFoundError:
    priority_df = pd.DataFrame(columns=["filename", "PRIO1"])

# Add priority points for the nearest neighbors
priority_df.loc[priority_df["filename"].isin(nearest_neighbor_filenames), "PRIO1"] = 1

# Save the updated priority DataFrame to the Excel file
priority_df.to_excel(priority_file_path, index=False)
