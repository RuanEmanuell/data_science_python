import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Lendo a base de dados
df = pd.read_csv("data/pokemon.csv")

# Visualizando primeiras linhas
print(df.head())

# Informações gerais
print(df.info())

# Removendo colunas que não ajudam muito
colunas_remover = ["name", "japanese_name", "classfication"]

for col in colunas_remover:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)


# Preenche colunas numéricas com mediana
df.fillna(df.median(numeric_only=True), inplace=True)

# Preenche colunas texto com Unknown
for col in df.select_dtypes(include="object").columns:
    df[col].fillna("Unknown", inplace=True)


# Convertendo colunas categóricas em números
le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Features (tudo será usado para clustering)
x = df.copy()


# Normalização dos dados (IMPORTANTE para K-Means)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

inertias = []

k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(x_scaled)
    inertias.append(kmeans.inertia_)

print("\n===== INÉRCIA (ELBOW METHOD) =====")
for k, inertia in zip(k_values, inertias):
    print("K:", k, "-> Inertia:", inertia)

# Escolha do K (baseado no elbow)
k_final = 4

kmeans_model = KMeans(n_clusters=k_final, random_state=42, n_init=10)

# Treinando modelo
labels = kmeans_model.fit_predict(x_scaled)

silhouette = silhouette_score(x_scaled, labels)

print("\n===== K-MEANS CLUSTERING =====")
print("Número de clusters:", k_final)
print("Silhouette Score:", silhouette)

# Resultados

df["cluster"] = labels

print("\nDistribuição dos clusters:")
print(df["cluster"].value_counts())


# O algoritmo K-Means agrupa os dados em clusters baseados em similaridade.
# Cada cluster representa um grupo de Pokémon com características semelhantes.

# O método do Elbow foi utilizado para estimar o melhor número de clusters,
# analisando o ponto onde a redução da inércia começa a estabilizar.

# O Silhouette Score mede a qualidade dos clusters:
# valores próximos de 1 indicam clusters bem separados
# valores próximos de 0 indicam sobreposição entre clusters

# Como não existe variável alvo neste problema, a avaliação é feita
# apenas por métricas internas de clustering.