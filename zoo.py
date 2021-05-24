# API CONSULTANDO A BASE DO OPENML = ZOO
from logging import error
from re import split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
from torch import nn
from torch import optim
from sklearn import datasets
import openml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import itertools
import json
import torch
import pymongo


# CODIGO COM ERRO
#datasets_df = openml.datasets.list_datasets(output_format="dataframe")
# print(datasets_df.head(n=10))

dataset = openml.datasets.get_dataset(62)
linhas, tipo, categorical_indicator, nome_das_colunas = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute)

# ACRECENTANDO EM UMA VARIAVEL
info = dataset.get_data(dataset_format="dataframe",
                        target=dataset.default_target_attribute)

# TRANFORMA EM DATAFRAME
df = pd.DataFrame(info[0])

# TROCANDO OS VALORES DO ZOO PARA PORTUGUES
a_renomear = {
    'hair': 'pelo',
    'feathers': 'penas',
    'eggs': 'poe_ovo',
    'milk': 'bebe_leite',
    'airborne': 'voador',
    'aquatic': 'aquatico',
    'predator': 'predador',
    'toothed': 'dentes',
    'backbone': 'vertebrado',
    'breathes': 'pulmoes',
    'venomous': 'venenoso',
    'fins': 'barbatanas',
    'legs': 'pernas',
    'tail': 'rabo',
    'domestic': 'domestico',
    'catsize': 'tamanho_de_gato'
}
# ATRIBUINDO OS VALORES TROCADOS
df = df.rename(columns=a_renomear)

# TROCANDO OS NOME DO DOS ANIMAIS PARA O VALOR ESPECIFICO DELE, E NA FRENTE A REF
for i in range(len(df)):
    if info[1][i] == 'mammal':
        df.loc[i, "tipo"] = 0  # mamifero
    if info[1][i] == 'fish':
        df.loc[i, "tipo"] = 1  # peixe
    if info[1][i] == 'reptile':
        df.loc[i, "tipo"] = 2  # reptil
    if info[1][i] == 'amphibian':
        df.loc[i, "tipo"] = 3  # anfibio
    if info[1][i] == 'insect':
        df.loc[i, "tipo"] = 4  # inseto
    if info[1][i] == 'invertebrate':
        df.loc[i, "tipo"] = 5  # invertebrado
    if info[1][i] == 'bird':
        df.loc[i, "tipo"] = 6  # passaro

# TROCANDO TODA TABELA DE VERDADEIRO PARA 1 E FALSO PARA 0
troca = {
    True: 1,
    False: 0,
}
df['pelo'] = df.pelo.map(troca)
df['penas'] = df.penas.map(troca)
df['poe_ovo'] = df.poe_ovo.map(troca)
df['bebe_leite'] = df.bebe_leite.map(troca)
df['voador'] = df.voador.map(troca)
df['aquatico'] = df.aquatico.map(troca)
df['predador'] = df.predador.map(troca)
df['dentes'] = df.dentes.map(troca)
df['vertebrado'] = df.vertebrado.map(troca)
df['pulmoes'] = df.pulmoes.map(troca)
df['venenoso'] = df.venenoso.map(troca)
df['barbatanas'] = df.barbatanas.map(troca)
df['rabo'] = df.rabo.map(troca)
df['domestico'] = df.domestico.map(troca)
df['tamanho_de_gato'] = df.tamanho_de_gato.map(troca)

titulos = df.columns

# TROCANDO OS VALORES DO ZOO PARA PORTUGUES
troca2 = {
    'mammal': 'mamifero',
    'fish': 'peixe',
    'reptile': 'reptil',
    'amphibian': 'anfibio',
    'insect': 'inseto',
    'invertebrate': 'invertebrado',
    'bird': 'passaro'
}

# JUNTANDO AS DUAS TABELAS PARA FICA EM UMA APENAS
dados = df.join(tipo)

dados['tipo'] = dados.type.map(troca2)

# VERIFICANDO A CUDA E CPU
args = {
    'epoch_num': 30,     # Número de épocas.
    'lr': 5e-5,           # Taxa de aprendizado.
    'weight_decay': 5e-4,  # Penalidade L2 (Regularização).
    'num_workers': 4,     # Número de threads do dataloader.
    'batch_size': 20,     # Tamanho do batch.
}

if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')

# print(args['device'])

torch.manual_seed(1)
indices = torch.randperm(len(df)).tolist()
treino_tam = int(0.8*len(df))
df_train = df.iloc[indices[:treino_tam]]
df_teste = df.iloc[indices[treino_tam:]]
df_train.to_csv('treinando.csv', index=False)
df_teste.to_csv('testando.csv', index=False)


class Verificador(Dataset):
    def __init__(self, csv, scaler_feat=None, scaler_label=None):

        self.dados = pd.read_csv(csv).to_numpy()

    def __getitem__(self, index):

        mostrar = self.dados[index][::]
        etiqueta = self.dados[index][-1:]

        # CONVERTE O TENSOR
        mostrar = torch.from_numpy(mostrar.astype(np.float32))
        etiqueta = torch.from_numpy(etiqueta.astype(np.float32))

        return mostrar, etiqueta

    def __len__(self):
        return len(self.dados)


dataset = Verificador('treinando.csv')
dado, rotulo = dataset[0]
train_set = Verificador('treinando.csv')
test_set = Verificador('testando.csv')

train_loader = DataLoader(train_set,
                          args['batch_size'],
                          num_workers=args['num_workers'],
                          shuffle=True)

test_loader = DataLoader(test_set,
                         args['batch_size'],
                         num_workers=args['num_workers'],
                         shuffle=False)


class ZOO(nn.Module):

    def __init__(self, entrada, oculto, saida):
        super(ZOO, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(entrada, oculto),
            nn.ReLU(),
            nn.Linear(oculto, oculto),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(oculto, saida),
            nn.ReLU(),
        )

    def forward(self, X):

        hidden = self.features(X)
        output = self.classifier(hidden)

        return output


entrada = train_set[0][0].size(0)
oculto = 11
saida = 1
net = ZOO(entrada, oculto, saida).to(args['device'])
criterio = nn.L1Loss().to(args['device'])
otimizar = optim.Adam(
    net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

testeX = torch.stack([tup[0] for tup in test_set])
testeX = testeX.to(args['device'])
testeY = torch.stack([tup[1] for tup in test_set])
prevY = net(testeX).cpu().data
data = torch.cat((testeY, prevY), axis=1)
df_results = pd.DataFrame(data, columns=['prevY', 'testeY'])
# print(df_results.head(5))


fig, ax = plt.subplots(5, 3, figsize=(20, 30), num=10)
sns.distplot(df.pelo, rug=True, rug_kws={"color": "g"},
             kde_kws={"color": "k", "lw": 3, "label": "pelo"},
             hist_kws={"histtype": "step", "linewidth": 3,
                       "alpha": 1, "color": "g"}, ax=ax[0, 0])
sns.distplot(df.penas, rug=True, rug_kws={"color": "g"},
             kde_kws={"color": "k", "lw": 3, "label": "penas"},
             hist_kws={"histtype": "step", "linewidth": 3,
                       "alpha": 1, "color": "g"},  ax=ax[0, 1])
sns.distplot(df.poe_ovo, rug=True, rug_kws={"color": "g"},
             kde_kws={"color": "k", "lw": 3, "label": "poe_ovo"},
             hist_kws={"histtype": "step", "linewidth": 3,
                       "alpha": 1, "color": "g"},  ax=ax[0, 2])
sns.distplot(df.bebe_leite, rug=True, rug_kws={"color": "g"},
             kde_kws={"color": "k", "lw": 3, "label": "bebe_leite"},
             hist_kws={"histtype": "step", "linewidth": 3,
                       "alpha": 1, "color": "g"},  ax=ax[1, 0])
sns.distplot(df.voador, rug=True, rug_kws={"color": "g"},
             kde_kws={"color": "k", "lw": 3, "label": "voador"},
             hist_kws={"histtype": "step", "linewidth": 3,
                       "alpha": 1, "color": "g"},  ax=ax[1, 1])
sns.distplot(df.predador, rug=True, rug_kws={"color": "g"},
             kde_kws={"color": "k", "lw": 3, "label": "predador"},
             hist_kws={"histtype": "step", "linewidth": 3,
                       "alpha": 1, "color": "g"},  ax=ax[1, 2])
sns.distplot(df.dentes, rug=True, rug_kws={"color": "g"},
             kde_kws={"color": "k", "lw": 3, "label": "dentes"},
             hist_kws={"histtype": "step", "linewidth": 3,
                       "alpha": 1, "color": "g"},  ax=ax[2, 0])
sns.distplot(df.vertebrado, rug=True, rug_kws={"color": "g"},
             kde_kws={"color": "k", "lw": 3, "label": "vertebrado"},
             hist_kws={"histtype": "step", "linewidth": 3,
                       "alpha": 1, "color": "g"},  ax=ax[2, 1])
sns.distplot(df.pulmoes, rug=True, rug_kws={"color": "g"},
             kde_kws={"color": "k", "lw": 3, "label": "pulmoes"},
             hist_kws={"histtype": "step", "linewidth": 3,
                       "alpha": 1, "color": "g"},  ax=ax[2, 2])
sns.distplot(df.venenoso, rug=True, rug_kws={"color": "g"},
             kde_kws={"color": "k", "lw": 3, "label": "venenoso"},
             hist_kws={"histtype": "step", "linewidth": 3,
                       "alpha": 1, "color": "g"},  ax=ax[3, 0])
sns.distplot(df.barbatanas, rug=True, rug_kws={"color": "g"},
             kde_kws={"color": "k", "lw": 3, "label": "barbatanas"},
             hist_kws={"histtype": "step", "linewidth": 3,
                       "alpha": 1, "color": "g"},  ax=ax[3, 1])
sns.distplot(df.rabo, rug=True, rug_kws={"color": "g"},
             kde_kws={"color": "k", "lw": 3, "label": "rabo"},
             hist_kws={"histtype": "step", "linewidth": 3,
                       "alpha": 1, "color": "g"},  ax=ax[3, 2])
sns.distplot(df.domestico, rug=True, rug_kws={"color": "g"},
             kde_kws={"color": "k", "lw": 3, "label": "domestico"},
             hist_kws={"histtype": "step", "linewidth": 3,
                       "alpha": 1, "color": "g"},  ax=ax[4, 0])
sns.distplot(df.tamanho_de_gato, rug=True, rug_kws={"color": "g"},
             kde_kws={"color": "k", "lw": 3, "label": "tamanho_de_gato"},
             hist_kws={"histtype": "step", "linewidth": 3,
                       "alpha": 1, "color": "g"}, ax=ax[4, 1])

plt.show()


x = df[['pelo', 'penas', 'poe_ovo', 'bebe_leite', 'voador', 'aquatico', 'predador', 'dentes',
        'vertebrado', 'pulmoes', 'venenoso', 'barbatanas', 'pernas', 'rabo', 'domestico', 'tamanho_de_gato']]
y = dados[['tipo']]


treino_x = x[:75]
treino_y = y[:75]
teste_x = x[75:]
teste_y = y[75:]

modelo = LinearSVC()
modelo.fit(treino_x, treino_y.values.ravel())

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.003)
previsoes = modelo.predict(teste_x)
accuracy_score(teste_y, previsoes)

animal = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]]
imprimir = modelo.predict(animal)[0]
print(f'Animal mais proximo foi ---> {imprimir.upper()} <---')

SEED = 5
np.random.seed(SEED)
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.20,
                                                        stratify=y)
print("Treinaremos com %d elementos e testaremos com %d elementos" %
      (len(treino_x), len(teste_x)))

modelo = LinearSVC()
modelo.fit(treino_x, treino_y.values.ravel())
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

print("================ Bem vindo ao treinando animais ================")
try:

    animal = []
    aux = []

    for indice in x:
        animal.append(int(input(
            f'Selecione a caracterificas do seu animal!\nTem {indice.upper()}\n1 - SIM\n0 - NAO ? ')))

    previsao = [animal]
    print('\n')
    print(f'Previsao : {previsao}')
    animalacuracia = modelo.predict(previsao)[0]
    print(
        f'Animal : {animalacuracia.upper()}')
    treino_x = x[:50]
    treino_y = y[:50]
    teste_x = x[50:]
    teste_y = y[50:]
    modelo = LinearSVC()
    modelo.fit(treino_x, treino_y.values.ravel())
    treino_x, teste_x, treino_y, teste_y = train_test_split(
        x, y, test_size=0.003)
    previsoes = modelo.predict(teste_x)
    accuracy_score(teste_y, previsoes)
    SEED = 5
    np.random.seed(SEED)
    treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.20,
                                                            stratify=y)
    print("Treinaremos com %d elementos e testaremos com %d elementos" %
          (len(treino_x), len(teste_x)))
    modelo = LinearSVC()
    modelo.fit(treino_x, treino_y.values.ravel())
    previsoes = modelo.predict(teste_x)
    acuracia = accuracy_score(teste_y, previsoes) * 100
    print("A acurácia foi %.2f%%" % acuracia)
    # pymongo usando o mongocliente
    cliente = pymongo.MongoClient(
        "mongodb+srv://admin:admin@cluster0.rrlac.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
    banco = cliente['zoo']  # buscando o nome do banco
    buscaanimais = banco["animais"]  # buscando a tebale do banco
    # print(cliente.list_database_names()) # consultando se está conectado no banco
    gravando = input('Deseja cadastrar o seus dados ex: sim/nao ? ')
    if gravando == 'sim':
        dados = {
            "Caracteristicas:": [{
                "Pelo":  animal[0],
                "Penas":  animal[1],
                "Poe_Ovo":  animal[2],
                "Bebe_Leite":  animal[3],
                "Voador":  animal[4],
                "Aquatico":  animal[5],
                "Predador":  animal[6],
                "Dentes":  animal[7],
                "Vertebrado":  animal[8],
                "Pulmoes":  animal[9],
                "Venenoso":  animal[10],
                "Barbatanas":  animal[11],
                "Pernas":  animal[12],
                "Rabo":  animal[13],
                "Domestico":  animal[14],
                "Tamanho_de_Gato":  animal[15],
            }],
            "Previsao": previsao,
            "Resultado": animalacuracia.upper(),
            "Treinando:": len(treino_x),
            "Testando:": len(teste_x),
            "Acuracia:": acuracia,
        }
        dadosgravado = buscaanimais.insert_one(dados)
        print(f'Obrigado por cadastrar seu dados em nosso sitemas!.... \n')
    else:
        print(f'Obrigado por usar nosso sitemas!.... \n')

except:
    print(error)
