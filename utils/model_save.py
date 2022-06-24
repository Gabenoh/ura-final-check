from settings.constants import TRAIN_CSV, VAL_CSV
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
from utils import DataLoader

# from sklearn.model_selection import GridSearchCV
# import sklearn as sk

data_train = pd.read_csv('../data/train.csv')
data_test = pd.read_csv('../data/val.csv')

train_data = DataLoader()
train_data.fit(data_train)
train_data = train_data.load_data()
test_data = DataLoader()
test_data.fit(data_train)
test_data = test_data.load_data()

y_data = train_data.Attrition_Flag
train_data = train_data.drop('Attrition_Flag', axis=1)
print('trein_data.shape:\n', train_data.shape)
y_true = test_data.Attrition_Flag
test_data = test_data.drop('Attrition_Flag', axis=1)
print('test data\n', np.array(test_data.iloc[0:5]))
print('true data\n', list(y_true.iloc[0:5]))
print('y_true\n', y_true.value_counts())

model = KNeighborsClassifier(n_neighbors=5)
model.fit(train_data, y_data)

y_pred = list(model.predict(test_data))
print('pred data\n', y_pred[0:5])
accuracy = accuracy_score(y_true, y_pred)
d = pd.DataFrame(y_pred)
print(d.value_counts())
print(accuracy)

with open('../models/KNN.pickle', 'wb') as f:
    pickle.dump(model, f)
