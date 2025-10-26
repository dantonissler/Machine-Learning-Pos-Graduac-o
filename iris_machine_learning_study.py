"""
Exemplo completo de aprendizado de máquina utilizando o conjunto de dados Íris.
Autor: Danton Rodrigues
Descrição:
    Este script demonstra conceitos fundamentais de Machine Learning,
    incluindo aprendizado supervisionado, não supervisionado e recomendações.
"""

# ==============================
# 1. Importação de bibliotecas
# ==============================
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 2. Carregamento e preparação do conjunto Íris
# ==========================================
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

print("\nAmostra dos dados originais:")
print(X.head())

# ==========================================
# 3. Normalização dos dados (redução de viés)
# ==========================================
# A normalização evita que atributos com escalas diferentes dominem o aprendizado.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 4. Divisão em treino e teste (Aprendizado supervisionado)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# ==========================================
# 5. Modelos de Classificação
# ==========================================

# 5.1. k-Nearest Neighbors (kNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# 5.2. Support Vector Machine (SVM)
svm = SVC(kernel="rbf", C=1, gamma="scale")
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# 5.3. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ==========================================
# 6. Avaliação dos modelos (supervisionado)
# ==========================================
print("\n========== MÉTRICAS DE CLASSIFICAÇÃO ==========")
for model_name, y_pred in [
    ("kNN", y_pred_knn),
    ("SVM", y_pred_svm),
    ("Random Forest", y_pred_rf)
]:
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModelo: {model_name}")
    print(f"Acurácia: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ==========================================
# 7. Aprendizado Não Supervisionado (Clustering + PCA)
# ==========================================

# 7.1. Redução de dimensionalidade (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 7.2. Clusterização com KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(X_scaled)

# 7.3. Métrica de qualidade (Silhouette Score)
sil_score = silhouette_score(X_scaled, clusters)
print(f"\nSilhouette Score (KMeans): {sil_score:.3f}")

# 7.4. Visualização dos clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", s=60)
plt.title("Clusterização (KMeans) com PCA - Dados Íris")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

# ==========================================
# 8. Sistema de Recomendação (Simples)
# ==========================================
# Exemplo: recomendar flores semelhantes com base nas medidas
df = X.copy()
df["species"] = y
df["species_name"] = df["species"].apply(lambda i: iris.target_names[i])

# Calcula similaridade entre amostras usando cosseno
similarity_matrix = cosine_similarity(X_scaled)

# Função de recomendação
def recomendar_similares(index, top_n=3):
    """
    Retorna as amostras mais semelhantes a uma flor específica.
    """
    scores = list(enumerate(similarity_matrix[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recomendacoes = [i for i, _ in scores[1:top_n+1]]
    return df.iloc[recomendacoes][["species_name"] + list(df.columns[:4])]

print("\n========== RECOMENDAÇÃO ==========")
idx = 0
print(f"\nFlor base: {df.iloc[idx]['species_name']}")
print(recomendar_similares(idx))

# ==========================================
# 9. Considerações sobre desafios em sistemas de recomendação
# ==========================================

print("\n========== CONCEITOS TEÓRICOS ==========")
print("""
• Início frio: resolvido combinando filtragem colaborativa (semelhança) com dados de conteúdo (atributos da flor).
• Viés: mitigado pela normalização e balanceamento da base.
• Diversidade: pode-se variar top_n ou penalizar itens populares.
• Escalabilidade: com grandes bases, usa-se amostragem ou técnicas de indexação.
• Privacidade: dados podem ser anonimizados ou criptografados.
""")

# ==========================================
# 10. Conclusão
# ==========================================
print("Demonstração completa finalizada com sucesso!")
