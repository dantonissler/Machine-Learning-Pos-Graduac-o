"""
03_agrupamento_kmeans.py
Agrupamento (clustering) com K-Means e visualização.
"""
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)

kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=60)
plt.title("Agrupamento K-Means (PCA reduzido)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
