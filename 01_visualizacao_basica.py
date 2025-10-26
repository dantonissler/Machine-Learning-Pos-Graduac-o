"""
01_visualizacao_basica.py
Entendimento e visualização do conjunto Íris
"""
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print(df.head())

# Visualização pairplot
sns.pairplot(df, hue='species', palette='viridis')
plt.suptitle("Visualização das características do conjunto Íris", y=1.02)
plt.show()
