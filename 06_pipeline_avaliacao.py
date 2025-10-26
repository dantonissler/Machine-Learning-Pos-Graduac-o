"""
06_pipeline_avaliacao.py
Criação de pipeline com normalização, modelo e avaliação cruzada.
"""
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

iris = load_iris()
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

scores = cross_val_score(pipeline, iris.data, iris.target, cv=5)
print("Acurácias por fold:", scores)
print("Acurácia média:", scores.mean())
