# Dataset utilizado: Labor. Esse Dataset foi usado para aprender as descrições de contratos aceitos e não aceitos
# Este programa acessa o DataSet, faz a tratativa dos dados, selecionando 2 de seus treasures,
# plota um grafico com os dados já tratados, depois treina esses dados e armazena os resultados no MongoDB

import numpy as np
import pymongo
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import openml
import pandas as pd
import matplotlib.pyplot as plt

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# buscando dataset: Labor https://www.openml.org/d/4
dataset = openml.datasets.get_dataset(4)
# printando informações do dataset

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# adicionando as informações do dataset a variável info
x, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# ajustando dados
# alterando tipo de y: Series para DataFrame
y_DF = pd.DataFrame(y)
# criando dados somente com as colunas desejadas
dados = x[["duration", "standby-pay"]]
# unindo y_DF com dados
dados = dados.join(y_DF)
# excluindo "standby-pay" = NaN
dados = dados.dropna()
print(dataset)
print(dados)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# gera o gráfico
sns.scatterplot(x="duration", y="standby-pay", hue="class", data=dados)
plt.show()

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# treinos e testes

# separando dados x e y
x = dados[["duration", "standby-pay"]]
# print(dados_x)
y = dados["class"]
y = pd.DataFrame(y)
# print(dados_y)

# #verificando tipos de dados raw_dados_x/raw_dados_y
# print(type(x))
# print('')
# print(type(y))

# definindo semente fixa para train_test_split e LinearSVC
SEED = 5
np.random.seed(SEED)

# algorítimo de treino
treino_x, teste_x, treino_y, teste_y = train_test_split(
    x, y, test_size=0.3, stratify=y)
print(
    f'Treinaremos com {len(treino_x)} e testaremos com {len(teste_x)} elementos')

# #rodando e treinando o modelo
modelo = LinearSVC()
modelo.fit(treino_x, treino_y.values.ravel())  # convertendo para uma dimensão
previsoes = modelo.predict(teste_x)
acuracia = accuracy_score(teste_y, previsoes) * 100
print(f'A acurácia foi de {acuracia:.2f} %')


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# inserindo valores no banco

# conectando ao servidor/conectando ao banco/criando coleção
myClient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myClient["labor"]
mycol = mydb["contratos"]

#inserindo dados no mongodb
dados.reset_index(inplace=True)
dados = dados.to_dict('records')
list_dados = (dados)
mycol.insert_many(list_dados)

#inserindo qtde de treinados/qtde de testados/acuracia no mongodb
dic = {"treinados": len(treino_x), "testados": len(
    teste_x), "acuracia": acuracia}
mycol.insert_one(dic)

# visualizando os valores
for x in mycol.find():
    print(x)

# # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
