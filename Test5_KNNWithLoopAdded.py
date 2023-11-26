# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:59:50 2023

@author: Danille
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
import pandas as pd

def visualize_knn(df, chosen_genre, k=10):
    # Filter data based on the chosen genre
    genre_data = df[df['label'] == chosen_genre]

    # Extract the desired columns for x, y, and labels
    x_column = "chroma_stft_mean"
    y_column = "chroma_stft_var"
    label_column = "label"
    data = df[[x_column, y_column, label_column]].values

    # Convert data to numeric type
    data[:, :2] = data[:, :2].astype(float)

    # Initialize K-Nearest Neighbors with k=
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data[:, :2])

    # Choose a random point from the selected genre
    random_point_index = random.randint(0, len(genre_data) - 1)
    random_point = genre_data.iloc[random_point_index][[x_column, y_column]].values.astype(float)

    # Print the random chosen filename
    print("Chosen Filename:", genre_data.iloc[random_point_index]["filename"])

    # Find the k-nearest neighbors for the random point
    distances, indices = nbrs.kneighbors(random_point.reshape(1, -1))

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

    # Plot the nearest neighbors with their original colors based on labels
    for neighbor_id in nearest_neighbor_ids:
        label_color = colors[np.where(labels == data[neighbor_id, 2])[0][0]]
        plt.scatter(data[neighbor_id, 0], data[neighbor_id, 1], color=label_color, marker='.', s=100)

    # Mark the chosen point with a black cross
    plt.scatter(random_point[0], random_point[1], color='black', marker='x', s=100, label='Chosen Point')

    plt.xlabel('X')
    plt.ylabel('Y')

    # Move legend outside the plot area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f'K-Nearest Neighbors Visualization with {x_column} and {y_column}')
    plt.show()

    # Print the filenames of the nearest neighbors
    nearest_neighbor_filenames = df.iloc[nearest_neighbor_ids]["filename"]
    print("Filenames of Nearest Neighbors:")
    print(nearest_neighbor_filenames)

    return nearest_neighbor_filenames

def main():
    # Load data from the CSV file
    file_path = r"C:\Users\Danille\Documents\minor\Block2\Tech\Test_KNearestNeighbors\Data\features_30_sec.csv"
    df = pd.read_csv(file_path)

    # Add priority points to a new Excel file
    priority_file_path = r"C:\Users\Danille\Documents\minor\Block2\Tech\Test_KNearestNeighbors\Data\DOC1_priotiteit.xlsx"

    # Read the existing priority file if it exists, or create a new DataFrame
    try:
        priority_df = pd.read_excel(priority_file_path)
    except FileNotFoundError:
        priority_df = pd.DataFrame(columns=["filename"])

    priority_counter = 1

    while True:
        # Ask the user to choose a genre
        chosen_genre = input("Choose a genre from: Blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock\n")

        # Visualize K-Nearest Neighbors
        nearest_neighbor_filenames = visualize_knn(df, chosen_genre)

        # Add priority points for the nearest neighbors to the new column
        priority_column_name = f"PRIO{priority_counter}"
        priority_df[priority_column_name] = 0
        priority_df.loc[priority_df["filename"].isin(nearest_neighbor_filenames), priority_column_name] = 1

        # Ask if the user wants to repeat the process or end
        user_choice = input("Do you want to do it again? (yes/no(if no, rerun the code for the next person)): ").lower()
        if user_choice != 'yes':
            break

        priority_counter += 1

    # Save the updated priority DataFrame to the Excel file
    priority_df.to_excel(priority_file_path, index=False)

if __name__ == "__main__":
    main()
