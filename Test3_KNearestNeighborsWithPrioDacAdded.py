# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 12:42:14 2023

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

# Extract the desired columns for x and y
x_column = "chroma_stft_mean"
y_column = "chroma_stft_var"
data = df[[x_column, y_column]].values

# Convert data to numeric type
data = data.astype(float)

# Initialize K-Nearest Neighbors with k=
k = 60
nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)

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

# Define the boundaries for the sections A, B, C, and D
x_A = (0, 1000)
y_A = (0, 1250)
x_B = (0, 1000)
y_B = (1251, 2500)
x_C = (1001, 2000)
y_C = (0, 1250)
x_D = (1001, 2000)
y_D = (1251, 2500)

# Create masks for the sections
mask_A = (data[:, 0] >= x_A[0]) & (data[:, 0] <= x_A[1]) & (data[:, 1] >= y_A[0]) & (data[:, 1] <= y_A[1])
mask_B = (data[:, 0] >= x_B[0]) & (data[:, 0] <= x_B[1]) & (data[:, 1] >= y_B[0]) & (data[:, 1] <= y_B[1])
mask_C = (data[:, 0] >= x_C[0]) & (data[:, 0] <= x_C[1]) & (data[:, 1] >= y_C[0]) & (data[:, 1] <= y_C[1])
mask_D = (data[:, 0] >= x_D[0]) & (data[:, 0] <= x_D[1]) & (data[:, 1] >= y_D[0]) & (data[:, 1] <= y_D[1])

# Plot the data points and the random example point along with its k-nearest neighbors
plt.figure(figsize=(8, 6))
plt.scatter(data[mask_A, 0], data[mask_A, 1], label='Section A', c='blue', alpha=0.6, marker='.')
plt.scatter(data[mask_B, 0], data[mask_B, 1], label='Section B', c='green', alpha=0.6, marker='.')
plt.scatter(data[mask_C, 0], data[mask_C, 1], label='Section C', c='red', alpha=0.6, marker='.')
plt.scatter(data[mask_D, 0], data[mask_D, 1], label='Section D', c='purple', alpha=0.6, marker='.')

# Plot the data points
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], label='Data Points', alpha=0.6, marker='.')

# Plot the random example point
plt.scatter(random_example_point[0, 0], random_example_point[0, 1], color='red', marker='x', s=100, label='Random Example Point')

# Increase the size of the nearest neighbors
plt.scatter(nearest_neighbor_coordinates[:, 0], nearest_neighbor_coordinates[:, 1], label='Nearest Neighbors', s=100)

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('K-Nearest Neighbors Visualization with Sections')
plt.show()

# Determine the category of the random example point
if x_A[0] <= random_example_point[0, 0] <= x_A[1] and y_A[0] <= random_example_point[0, 1] <= y_A[1]:
    category = "Section A"
elif x_B[0] <= random_example_point[0, 0] <= x_B[1] and y_B[0] <= random_example_point[0, 1] <= y_B[1]:
    category = "Section B"
elif x_C[0] <= random_example_point[0, 0] <= x_C[1] and y_C[0] <= random_example_point[0, 1] <= y_C[1]:
    category = "Section C"
elif x_D[0] <= random_example_point[0, 0] <= x_D[1] and y_D[0] <= random_example_point[0, 1] <= y_D[1]:
    category = "Section D"
else:
    category = "Uncategorized"

# Print the category of the random example point
print("Category of Random Example Point:", category)

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