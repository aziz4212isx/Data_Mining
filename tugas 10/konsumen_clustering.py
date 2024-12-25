import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data from konsumen.csv
data = pd.read_csv(r"D:\SEMESTER 5\Data Mining\tugas 10\tugas 10\konsumen.csv")

# Standardize the data (if necessary)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Apply K-Means clustering
num_clusters = 3  # Ubah jumlah cluster jika diperlukan
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(scaled_data)
labels = kmeans.labels_
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# Add cluster labels to the dataset
data['Cluster'] = labels

# Save the clustered data to a new CSV file
data.to_csv("konsumen_clustered.csv", index=False)

# Plot the data (using first two columns for visualization)
plt.figure(figsize=(8, 6))
for i in range(num_clusters):
    cluster_points = data[data['Cluster'] == i]
    plt.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1], label=f"Cluster {i+1}")

plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label="Centroids")
plt.title("K-Means Clustering on konsumen.csv")
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()
