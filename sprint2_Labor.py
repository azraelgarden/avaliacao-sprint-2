#Dataset utilizado: Labor
#Esse Dataset foi usado para aprender as descrições de contratos aceitos e não aceitos


import pymongo
from scipy.sparse import data
from sklearn import datasets
import sklearn
from sklearn.svm import LinearSVC
import seaborn as sns
import openml
import pandas as pd
import matplotlib.pyplot as plt

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#conectando ao servidor/conectando ao banco/criando coleção
myClient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myClient["labor"]
myCol = mydb["contratos"]

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#buscando dataset: Labor https://www.openml.org/d/4
dataset = openml.datasets.get_dataset(4)
#printando informações do dataset
#print(dataset)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#adicionando as informações do dataset a variável info
x, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

#isolando coluna standby-py
standby_pay = x['standby-pay']
#print(standby_pay)
#isolando coluna duration
duration = x['duration']
#print(duration)
#isolando classe
classe = y
#print(classe)
print(classe.shape)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# #cria o gráfico
sns.scatterplot(x=standby_pay, y=duration, hue=classe, data=x)
# #imprime o gráfico
# #plt.show()
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=









