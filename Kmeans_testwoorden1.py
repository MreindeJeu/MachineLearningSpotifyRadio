# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:45:35 2023

@author: Danille
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Lijsten met tekstwaarden voor de X- en Y-assen
x = ["Rock", "Pop", "Jazz", "Hip-Hop", "Classical", "Electronic", "Reggae", "Country", "R&B", "Blues"]
y = ["Energetic", "Energetic", "Relaxing", "Energetic", "Relaxing", "Energetic", "Relaxing", "Energetic", "Energetic", "Relaxing"]

# Maak een numerieke representatie voor de tekstwaarden
x_numerical = list(range(len(x)))
y_numerical = list(range(len(y)))

data = list(zip(x_numerical, y_numerical))

kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

plt.scatter(x_numerical, y_numerical, c=kmeans.labels_)

# Voeg omschrijvingen toe aan de X- en Y-assen met tekstwaarden
plt.xticks(x_numerical, x, rotation=45)
plt.yticks(y_numerical, y)
plt.xlabel("Music Genre")
plt.ylabel("Mood")

plt.show()
