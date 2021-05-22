from dataset import Datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

SEED = 46
np.random.seed(SEED)


dataset = Datasets(925)
x, y, categorical_indicator, attribute_names = dataset.get()

y = pd.DataFrame(y)

data = x[['northsouth','eastwest']]
print(data)

data['class'] = y
# print(data)

# sns.scatterplot(x="eastwest", y="northsouth", hue="class", data=data)
# plt.show()

x = data[['northsouth','eastwest']]
y = pd.DataFrame(data['class'])

print(f'{type(x)}\n {type(y)}')

train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.3, stratify=y)

print(train_y)

print(f'Train: {len(train_x)} \n test: {len(test_x)}')

model = LinearSVC(dual=False)
model.fit(train_x, train_y.values.ravel())
predict = model.predict(train_x)
accuracy = accuracy_score(train_y, predict) * 100
print(f'accuracy: {accuracy:.2f}%')