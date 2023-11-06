# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:25:17 2023

@author: Danille
"""
#https://www.w3schools.com/python/python_ml_k-means.asp

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.show()

