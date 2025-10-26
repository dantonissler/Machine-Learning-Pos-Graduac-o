"""
07_explicabilidade_otimizacao.py
Otimização com GridSearchCV e análise de importância de features.
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

iris = load_iris()
param_grid = {"n_estimators": [50, 100, 200], "max_depth": [2, 4, 6, None]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(iris.data, iris.target)

print("Melhores parâmetros:", grid.best_params_)
print("Melhor acurácia:", grid.best_score_)

# Importância das features
model = grid.best_estimator_
importances = model.feature_importances_
features = pd.Series(importances, index=iris.feature_names)
features.sort_values().plot(kind="barh", title="Importância das Features")
plt.show()
