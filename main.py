import numpy as np
import pymongo
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import openml
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")

client = MongoClient('localhost', 27017)
client.list_database_names()

mydb = client["Cleve"]
mycol = mydb["Dados"]

dataset = openml.datasets.get_dataset(40710)

x, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute)

info = dataset.get_data(dataset_format = "dataframe", target = dataset.default_target_attribute)
df = pd.DataFrame(info[0])

y_DF = pd.DataFrame(y)

dados = x[["Age", "Max_heart_rate"]]

dados['class'] = y_DF

dados = dados.dropna()
print(dados)

sns.scatterplot(x="Age", y="Max_heart_rate", hue="class", data=dados)
plt.show()

X = dados[["Age", "Max_heart_rate"]]
y = dados["class"]

SEED = 11
np.random.seed(SEED)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, stratify=y)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)
pca = PCA(n_components=None,random_state=0)
X_train = pca.fit_transform(X_train)
X_test =pca.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
lr_score = lr.score(X_test,y_test)
sv = SVC(kernel ='rbf',random_state=0)
sv.fit(X_train,y_train)
sv_pred = sv.predict(X_test)
sv_score = sv.score(X_test,y_test)
rf_regressor = RandomForestClassifier(n_estimators = 1000, random_state = 0)
rf_regressor.fit(X_train, y_train)
rf_pred = rf_regressor.predict(X_test)
rf_score = rf_regressor.score(X_test,y_test)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
knn_score = knn.score(X_test,y_test)
nv = GaussianNB()
nv.fit(X_train,y_train)
nv_sc = nv.score(X_test,y_test)
print(f'Usamos {len(X_train)} para treino e {len(X_test)} de elementos para teste')


modelo = LinearSVC()
modelo.fit(X_train, y_train.values.ravel())
previsoes = modelo.predict(X_test)
acuracia = accuracy_score(y_test, previsoes) * 100
print(f'A acurácia foi de {acuracia:.2f} %')

dic = {
        "Teste": len(X_train),
        "Treino": len(X_train), 
        "Acuracia": acuracia, 
        "Lr_score": lr_score,
        "Nv_sc": nv_sc,
        "Rf_score": rf_score,
        "Sv_score": sv_score
    }
mycol.insert_one(dic)

for dado in mycol.find():
  print(dado)