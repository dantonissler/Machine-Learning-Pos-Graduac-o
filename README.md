```markdown
# 🤖 Aprendizado de Máquina na Prática — Projeto Progressivo com Python  

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Ativo-success.svg)

---

## 🧠 Visão Geral

Este repositório apresenta uma **série de algoritmos progressivos de aprendizado de máquina (Machine Learning)**, mostrando passo a passo a evolução desde o entendimento dos dados até a otimização de modelos e explicabilidade.  

O objetivo é servir como um **guia de estudos prático**, com exemplos bem comentados, prontos para execução, cobrindo desde conceitos básicos até técnicas intermediárias e avançadas.

---

## 🗂️ Estrutura do Projeto

```

machine_learning_progressive/
│
├── 01_visualizacao_basica.py
├── 02_classificacao_supervisionada.py
├── 03_agrupamento_kmeans.py
├── 04_regressao_linear.py
├── 05_sistema_recomendacao.py
├── 06_pipeline_avaliacao.py
├── 07_explicabilidade_otimizacao.py
├── README.md
└── requirements.txt

````

---

## 🚀 Execução do Projeto

### 1️⃣ Criar o ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate     # (Linux/Mac)
.venv\Scripts\activate        # (Windows)
````

### 2️⃣ Instalar as dependências

Crie um arquivo `requirements.txt` com o conteúdo abaixo:

```txt
scikit-learn
pandas
numpy
matplotlib
seaborn
```

E instale:

```bash
pip install -r requirements.txt
```

### 3️⃣ Executar os scripts

Cada script é independente.
Por exemplo, para rodar o nível 2 (classificação):

```bash
python 02_classificacao_supervisionada.py
```

---

## 🔹 Nível 1 — Visualização e Exploração de Dados

**Arquivo:** `01_visualizacao_basica.py`

Aprende-se a **carregar, entender e visualizar** dados com `matplotlib` e `seaborn`.
O conjunto usado é o clássico **Íris**, amplamente adotado em experimentos de ML.

> 🎯 **Objetivo:** Compreender padrões iniciais e relações entre variáveis.

---

## 🔹 Nível 2 — Classificação Supervisionada

**Arquivo:** `02_classificacao_supervisionada.py`

Demonstra o uso de um **modelo supervisionado (k-Nearest Neighbors)**, que aprende com dados rotulados para prever classes futuras.

> 🧩 Conceitos:
>
> * Treinamento e teste (`train_test_split`)
> * Normalização (`StandardScaler`)
> * Avaliação de desempenho (`accuracy`, `classification_report`)

---

## 🔹 Nível 3 — Agrupamento (Aprendizado Não Supervisionado)

**Arquivo:** `03_agrupamento_kmeans.py`

Implementa o **K-Means** para agrupar amostras sem rótulos e utiliza **PCA** para reduzir a dimensionalidade e visualizar os grupos.

> 🧩 Conceitos:
>
> * Clusterização (sem rótulos)
> * Redução de dimensionalidade (PCA)
> * Visualização 2D

---

## 🔹 Nível 4 — Regressão Linear

**Arquivo:** `04_regressao_linear.py`

Aplica **regressão linear** ao dataset *California Housing*, prevendo valores contínuos (preço de casas).

> 🧩 Conceitos:
>
> * Modelos de regressão
> * Métricas MAE (erro absoluto médio) e R²
> * Avaliação de previsões

---

## 🔹 Nível 5 — Sistema de Recomendação

**Arquivo:** `05_sistema_recomendacao.py`

Cria um **sistema de recomendação baseado em similaridade de atributos** (conteúdo).
Utiliza a **similaridade do cosseno** para recomendar itens semelhantes.

> 🧩 Conceitos:
>
> * Recomendação baseada em conteúdo
> * Similaridade de cosseno
> * Combate ao “problema do início frio” (novos usuários/itens)

---

## 🔹 Nível 6 — Pipeline e Avaliação Cruzada

**Arquivo:** `06_pipeline_avaliacao.py`

Mostra como integrar etapas de pré-processamento e modelagem em um único **pipeline**.
Aplica **validação cruzada (cross-validation)** para aumentar a confiabilidade.

> 🧩 Conceitos:
>
> * Pipelines (`Pipeline`)
> * Validação cruzada (`cross_val_score`)
> * Random Forest como classificador robusto

---

## 🔹 Nível 7 — Explicabilidade e Otimização

**Arquivo:** `07_explicabilidade_otimizacao.py`

Realiza **otimização de hiperparâmetros** com `GridSearchCV` e gera um gráfico de **importância das variáveis**.

> 🧩 Conceitos:
>
> * Busca de hiperparâmetros (`GridSearchCV`)
> * Importância de features (feature importance)
> * Interpretação do modelo

---

## ⚙️ Conceitos Fundamentais

| Conceito                           | Descrição                                                              |
| ---------------------------------- | ---------------------------------------------------------------------- |
| **Aprendizado supervisionado**     | Modelos aprendem com dados rotulados (ex.: classificação).             |
| **Aprendizado não supervisionado** | Descobre padrões e grupos ocultos sem rótulos.                         |
| **Regressão**                      | Predição de valores contínuos (ex.: preços).                           |
| **Recomendação**                   | Sugere itens com base em semelhanças.                                  |
| **Pipeline**                       | Encadeia etapas de pré-processamento e modelagem.                      |
| **Otimização de hiperparâmetros**  | Ajusta automaticamente os parâmetros do modelo para melhor desempenho. |
| **Explicabilidade**                | Entender por que o modelo toma certas decisões.                        |

---

## 📈 Exemplo de Saída (resumida)

```
Acurácia (kNN): 0.97
              precision    recall  f1-score   support
setosa           1.00      1.00      1.00        15
versicolor       0.94      1.00      0.97        15
virginica        1.00      0.93      0.96        15
```

E gráfico de clusters (Nível 3):

![Clusters](imgs/clusters.png)

---

## 🧩 Próximos Passos

Sugestões de evolução do projeto:

* 🔸 Implementar redes neurais com **TensorFlow / PyTorch**
* 🔸 Criar API de predição com **FastAPI**
* 🔸 Implantar modelo com **Streamlit** ou **Azure ML**
* 🔸 Adicionar **explicabilidade SHAP/LIME**

---

## 📚 Referências

* [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
* [Python Official Docs](https://docs.python.org/3/)
* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)

---

## 👨‍💻 Autor

**Danton Rodrigues**
💼 *Estudos em Inteligência Artificial, Aprendizado de Máquina e Automação Judicial*
📫 [LinkedIn](https://www.linkedin.com/) • [GitHub](https://github.com/dantonissler)

---

## 📝 Licença

Este projeto está licenciado sob os termos da **MIT License**.
Sinta-se à vontade para estudar, adaptar e expandir este conteúdo.

---

> *"Machine Learning é um processo contínuo de aprendizado — tanto para as máquinas quanto para quem as ensina."*
> — **Danton Rodrigues**

```

---

Deseja que eu gere automaticamente este `README.md` e os **7 scripts `.py`** organizados em pastas, prontos para baixar como `.zip` e subir diretamente no GitHub?  
Posso incluir também um `requirements.txt` e um banner ASCII para o início do README.
```
