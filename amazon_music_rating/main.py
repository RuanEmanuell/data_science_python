import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Lendo a base de dados
df = pd.read_csv("data/amazon_music.csv")

# Visualizando primeiras linhas
print(df.head())

# Informações gerais
print(df.info())

# Removendo colunas que não ajudam muito
colunas_remover = ["review", "reviewTime", "helpful", "nothelpful"]

for col in colunas_remover:
	if col in df.columns:
		df.drop(col, axis=1, inplace=True)


# Preenche colunas numéricas com mediana
df.fillna(df.median(numeric_only=True), inplace=True)

# Preenche colunas texto com Unknown
for col in df.select_dtypes(include=["object", "string"]).columns:
	df[col] = df[col].fillna("Unknown")


# Garante que rating é numérico
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["rating"] = df["rating"].fillna(df["rating"].median())

# Features principais para filtragem colaborativa
dados_cf = df[["uid", "pid", "rating"]].copy()


# Filtra usuários e itens com poucas interações
min_interacoes_usuario = 5
min_interacoes_item = 5

usuarios_validos = dados_cf["uid"].value_counts()
itens_validos = dados_cf["pid"].value_counts()

dados_cf = dados_cf[dados_cf["uid"].isin(usuarios_validos[usuarios_validos >= min_interacoes_usuario].index)]
dados_cf = dados_cf[dados_cf["pid"].isin(itens_validos[itens_validos >= min_interacoes_item].index)]

print("\n===== BASE APÓS FILTROS =====")
print("Linhas:", len(dados_cf))
print("Usuários únicos:", dados_cf["uid"].nunique())
print("Itens únicos:", dados_cf["pid"].nunique())


# Divisão treino e teste
train_data, test_data = train_test_split(dados_cf, test_size=0.2, random_state=42)

# Matriz usuário-item para o treinamento
user_item_matrix = train_data.pivot_table(index="uid", columns="pid", values="rating", aggfunc="mean")
user_item_filled = user_item_matrix.fillna(0).astype(np.float32)


# Modelo de Filtragem Colaborativa baseado em usuários (KNN)
k_neighbors = 10

modelo_knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k_neighbors + 1)
modelo_knn.fit(user_item_filled.values)

user_to_index = {user_id: idx for idx, user_id in enumerate(user_item_filled.index)}
pid_to_index = {pid: idx for idx, pid in enumerate(user_item_filled.columns)}
global_mean = train_data["rating"].mean()
matriz_usuario_item = user_item_filled.values

# Pré-cálculo dos vizinhos para acelerar as previsões
distancias_vizinhos, indices_vizinhos = modelo_knn.kneighbors(matriz_usuario_item, n_neighbors=k_neighbors + 1)


def prever_rating(uid, pid):
	if uid not in user_to_index:
		return float(global_mean)

	user_idx = user_to_index[uid]
	user_vector = matriz_usuario_item[user_idx]

	if pid not in pid_to_index:
		ratings_usuario = user_vector[user_vector > 0]
		if len(ratings_usuario) > 0:
			return float(ratings_usuario.mean())
		return float(global_mean)

	pid_idx = pid_to_index[pid]
	distances = distancias_vizinhos[user_idx]
	indices = indices_vizinhos[user_idx]

	soma_ponderada = 0.0
	soma_similaridade = 0.0

	for distancia, vizinho_idx in zip(distances, indices):
		if vizinho_idx == user_idx:
			continue

		similaridade = 1 - distancia
		rating_vizinho = matriz_usuario_item[vizinho_idx, pid_idx]

		if rating_vizinho > 0:
			soma_ponderada += similaridade * rating_vizinho
			soma_similaridade += similaridade

	if soma_similaridade > 0:
		predicao = soma_ponderada / soma_similaridade
	else:
		ratings_usuario = user_vector[user_vector > 0]
		if len(ratings_usuario) > 0:
			predicao = ratings_usuario.mean()
		else:
			predicao = global_mean

	return float(np.clip(predicao, 1.0, 5.0))


# Avaliação do sistema no conjunto de teste
test_sample = test_data.sample(n=min(2000, len(test_data)), random_state=42)

y_true = []
y_pred = []

for _, row in test_sample.iterrows():
	uid = row["uid"]
	pid = row["pid"]
	rating_real = row["rating"]

	rating_previsto = prever_rating(uid, pid)

	y_true.append(rating_real)
	y_pred.append(rating_previsto)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("\n===== FILTRAGEM COLABORATIVA (USER-BASED KNN) =====")
print("Tamanho treino:", len(train_data))
print("Tamanho teste:", len(test_data))
print("Amostra avaliada:", len(test_sample))
print("MAE:", mae)
print("RMSE:", rmse)


# Geração de recomendações para um usuário exemplo
usuario_exemplo = train_data["uid"].value_counts().idxmax()

itens_avaliados = set(train_data[train_data["uid"] == usuario_exemplo]["pid"])
todos_itens = set(user_item_filled.columns)
itens_nao_avaliados = list(todos_itens - itens_avaliados)

recomendacoes = []

for pid in itens_nao_avaliados:
	score = prever_rating(usuario_exemplo, pid)
	recomendacoes.append((pid, score))

top_10 = sorted(recomendacoes, key=lambda x: x[1], reverse=True)[:10]

print("\nTop 10 recomendações para o usuário:", usuario_exemplo)
for pid, score in top_10:
	print("Item:", pid, "-> Score previsto:", round(score, 3))


# O algoritmo de Filtragem Colaborativa utiliza as interações de usuários
# para recomendar itens com base em padrões de similaridade.

# Neste projeto foi utilizado o KNN baseado em usuários,
# medindo semelhança por distância cosseno entre vetores usuário-item.

# A avaliação do sistema foi feita com MAE e RMSE:
# MAE mede o erro absoluto médio das previsões
# RMSE penaliza mais fortemente erros maiores

# Como não existe uma variável alvo de classificação tradicional,
# a avaliação considera a proximidade entre notas reais e previstas.