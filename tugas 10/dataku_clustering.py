import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = np.loadtxt("D:/SEMESTER 5/Data Mining/tugas 10/tugas 10/dataku.txt")


# Apply K-Means clustering
num_clusters = 3  # Ubah jumlah cluster jika diperlukan
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the data and centroids
plt.figure(figsize=(8, 6))
for i in range(num_clusters):
    cluster_points = data[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i+1}")

plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label="Centroids")
plt.title("K-Means Clustering on dataku.txt")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()
