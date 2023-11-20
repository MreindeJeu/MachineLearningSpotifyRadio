import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Functie voor het berekenen van lineaire interpolatie tussen twee kleuren
def lerp_color(color1, color2, weight):
    return (1 - weight) * np.array(color1) + weight * np.array(color2)

# Functie om het scatter plot bij te werken op basis van de schuifregelaar
def update(val):
    size_val = size_slider.val
    alpha_val = alpha_slider.val

    plt.cla()  # Wis het vorige plot bij verandering
    for i in range(len(X)):
        punt = X[i]
        afstanden = np.linalg.norm(punt - centroids, axis=1)  # Euclidische afstanden tot de centroids
        dichtstbijzijnde_clusters = np.argsort(afstanden)[:2]  # Indices van de twee dichtstbijzijnde clusters
        gewichten = 1 - afstanden[dichtstbijzijnde_clusters] / afstanden[dichtstbijzijnde_clusters].sum()  # Gewichten voor lineaire interpolatie
        gemengde_kleur = lerp_color(cluster_colors[dichtstbijzijnde_clusters[0]], cluster_colors[dichtstbijzijnde_clusters[1]], gewichten[0])
        plt.scatter(punt[0], punt[1], c=[gemengde_kleur], s=size_val, alpha=alpha_val, edgecolors='k')

    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroiden')
    plt.title('K-Means Clusters met Gradient Kleuren')
    plt.xlabel('Kenmerk 1')
    plt.ylabel('Kenmerk 2')
    plt.legend()
    plt.draw()

# Genereer synthetische data
X, y = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=2.0)

# K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Viond cluster centrum
centroids = kmeans.cluster_centers_

# Definieer kleuren voor elk cluster
cluster_colors = {
    0: (1, 0, 0),    # Rood
    1: (1, 1, 0),    # Geel
    2: (0, 0, 1),    # Blauw
    3: (0, 1, 0)     # Groen
}

# Maak een figuur en as
fig, ax = plt.subplots(figsize=(10, 6))

# Plot scatter plot
for i in range(len(X)):
    punt = X[i]
    afstanden = np.linalg.norm(punt - centroids, axis=1)  # Euclidische afstanden tot de centroids
    dichtstbijzijnde_clusters = np.argsort(afstanden)[:2]  # Indices van de twee dichtstbijzijnde clusters
    gewichten = 1 - afstanden[dichtstbijzijnde_clusters] / afstanden[dichtstbijzijnde_clusters].sum()  # Gewichten voor lineaire interpolatie
    gemengde_kleur = lerp_color(cluster_colors[dichtstbijzijnde_clusters[0]], cluster_colors[dichtstbijzijnde_clusters[1]], gewichten[0])
    ax.scatter(punt[0], punt[1], c=[gemengde_kleur], s=150, alpha=0.2, edgecolors='k')

ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroiden')
ax.set_title('K-Means Clusters met gebruik van gemengde Kleuren')
ax.set_xlabel('Kenmerk 1')
ax.set_ylabel('Kenmerk 2')
ax.legend()

# Voeg schuifregelaars toe voor grootte en transparantie
size_slider_ax = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
alpha_slider_ax = plt.axes([0.1, 0.06, 0.65, 0.03], facecolor='lightgoldenrodyellow')

size_slider = Slider(size_slider_ax, 'Punt Grootte', 1, 300, valinit=150, valstep=1)
alpha_slider = Slider(alpha_slider_ax, 'Transparantie', 0.01, 1.0, valinit=0.2, valstep=0.01)

# Koppel de update functie aan de schuifregelaars
size_slider.on_changed(update)
alpha_slider.on_changed(update)

plt.show()
