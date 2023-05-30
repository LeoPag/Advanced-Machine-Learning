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
from sklearn.model_selection import GridSearchCV

np.random.seed(2345)

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

max_depth = [1, 3, 5, 10, 15]
n_estimators = [500, 750, 1000,  1250,  1500, 1750, 2000]
min_child_weight = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
colsample_bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
alpha = [0.01, 0.1, 1, 10]
eta = [0.01, 0.1, 0.2, 0.3]
gamma = [1, 3, 5, 7, 10, 13, 15,  17, 20, 30, 50, 100]

def hyperParameterTuning(X_train, y_train):
    param_tuning = {
        'max_depth': [3, 5],
        'n_estimators': [500, 1000],
        'min_child_weight': [3, 7, 10],
        'subsample': [0.5, 0.7, 0.8],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
        'alpha' : [0.1],
        'eta' : [0.01],
        'gamma' : [3, 7, 10, 15, 20, 30],
        'objective': ['reg:squarederror', 'reg:logistic']
    }

    xgb_model = xgb.XGBRegressor()

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,
                           #scoring = 'neg_mean_absolute_error', #MAE
                           #scoring = 'neg_mean_squared_error',  #MSE
                           cv = 5,
                           n_jobs = -1,
                           verbose = 1)

    gsearch.fit(X_train,y_train)

    return gsearch.best_params_

hyperParameterTuning(X_train, y_train)
