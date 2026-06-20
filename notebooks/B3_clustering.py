import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("../data/corpus.csv")
embeddings = np.load("../data/embeddings.npy")

Path("../figures").mkdir(exist_ok=True)
Path("../data/clustering").mkdir(parents=True, exist_ok=True)

# PCA to 2D
pca = PCA(n_components=2, random_state=42)
points_2d = pca.fit_transform(embeddings)

# KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(embeddings)

df["cluster"] = clusters
df["pca_x"] = points_2d[:, 0]
df["pca_y"] = points_2d[:, 1]

score = silhouette_score(embeddings, clusters)
print("Silhouette score:", score)

df[["song_id", "title", "year", "album", "cluster", "pca_x", "pca_y"]].to_csv(
    "../data/clustering/song_clusters.csv",
    index=False,
    encoding="utf-8-sig"
)

# Plot by cluster
plt.figure(figsize=(10, 7))
plt.scatter(df["pca_x"], df["pca_y"], c=df["cluster"])
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Song Embeddings Clustering - KMeans k=3")
plt.savefig("../figures/B3_pca_clusters.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot by year
plt.figure(figsize=(10, 7))
plt.scatter(df["pca_x"], df["pca_y"], c=df["year"])
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Song Embeddings PCA - Colored by Year")
plt.colorbar(label="Year")
plt.savefig("../figures/B3_pca_years.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved:")
print("../data/clustering/song_clusters.csv")
print("../figures/B3_pca_clusters.png")
print("../figures/B3_pca_years.png")

print("\nCluster counts:")
print(df["cluster"].value_counts())

print("\nAverage year by cluster:")
print(df.groupby("cluster")["year"].mean())