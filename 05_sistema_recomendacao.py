"""
05_sistema_recomendacao.py
Recomendação simples baseada em similaridade de cosseno.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

sim_matrix = cosine_similarity(df.iloc[:, :-1])

def recomendar(index, n=3):
    similares = list(enumerate(sim_matrix[index]))
    similares = sorted(similares, key=lambda x: x[1], reverse=True)
    top = [i for i, _ in similares[1:n+1]]
    return df.iloc[top]

print("Flor base:")
print(df.iloc[0])
print("\nRecomendações semelhantes:")
print(recomendar(0))
