import numpy as np
import csv
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, make_scorer
from numpy import mean
from numpy import std

np.random.seed(3232)

# -------------------------  GET THE DATA --------------------------------
X = pd.read_csv('X_train.csv', float_precision='high').drop('id', axis=1)
y = pd.read_csv('y_train.csv', float_precision='high').drop('id', axis=1)

# X_new = X
# fill the missing values with the median
X = X.fillna(X.mean())

# -------------------------  GET THE OUTLIERS  --------------------------------
model = IsolationForest(contamination = 0.01, n_estimators = 1000)
model.fit(X)
anomalies = model.predict(X)
# X_new = X_new[anomalies == 1]
X = X[anomalies == 1]
y = y[anomalies == 1]

# X_new = X_new.fillna(X_new.median())
# X = X_new
print(np.array(X).shape)
# -------------------------  INPUT REDUCTION  --------------------------------
correlation_threshold = 0.1

Xprova = X
Xprova["y"] = y
correlation = (abs(Xprova.corr()["y"]))

selectedfeatures = abs(Xprova.corr()["y"][abs(Xprova.corr()["y"]) > correlation_threshold].drop("y")).index.tolist()
X = X[selectedfeatures]

# --------------- SPLIT IN TRAIN, TEST AND VALIDATION ---------------------
# # split X in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

# -------------------------  NORMALIZE  --------------------------------
from sklearn.preprocessing import MinMaxScaler
# define min max scaler
scaler = MinMaxScaler()

max_X_train = np.amax(np.array(X_train), 0)
min_X_train = np.amin(np.array(X_train), 0)
# transform data
X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)
X_test = np.array(X_test)
for i in range(X_test.shape[1]):
    column = X_test[:, i]
    column_std = (column - min_X_train[i]) / (max_X_train[i] - min_X_train[i])
    # column_scaled = column_std * (max_X_train[i] - min_X_train[i]) + min_X_train[i]
    X_test[:, i] = column_std

# --------------- CREATE THE MODEL ---------------------
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import *
import sklearn.gaussian_process as gp

kernel = RBF(length_scale=1.25)
regressor = SVR(kernel = kernel, epsilon = 0.01, C=100)

# define model evaluation method
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# # evaluate model
# r2_scorer = make_scorer(r2_score)
# SVRscores = cross_val_score(regressor, X, y, scoring = r2_scorer, cv=cv, n_jobs=-1)
# print('Mean MAE: %.3f (%.3f)' % (mean(SVRscores), std(SVRscores)))

_ = regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
score = r2_score(np.array(y_test), y_pred)
print(f"The score is: {score}")




# ------------- PREDICT ON THE TEST REAL TEST DATA -------------------
regressor = SVR(kernel = kernel, epsilon = 0.01, C=100)
defscaler = MinMaxScaler()
max_X_train = np.amax(np.array(X), 0)
# print(max_X_train)
min_X_train = np.amin(np.array(X), 0)
X = defscaler.fit_transform(X)
_ = regressor.fit(X, y)

X_test = pd.read_csv('X_test.csv', float_precision='high').drop('id', axis=1)
# fill the missing values with the median
X_test = X_test.fillna(X_test.median())
X_test = X_test[selectedfeatures]


X_test = np.array(X_test)
for i in range(X_test.shape[1]):
    column = X_test[:, i]
    column_std = (column - min_X_train[i]) / (max_X_train[i] - min_X_train[i])
    # column_scaled = column_std * (max_X_train[i] - min_X_train[i]) + min_X_train[i]
    X_test[:, i] = column_std

# print(X_test)
y_pred = regressor.predict(X_test)

f = open("output_svr.csv", "w")
f.write("id,y\n")
for i in range(len(y_pred)):
    f.write(str(float(i)) + "," + str(y_pred[i]) + "\n")
f.close()
