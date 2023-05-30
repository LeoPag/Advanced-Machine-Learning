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

# xgbr = xgb.XGBRegressor (max_depth = 10,
#                                             n_estimators = 1000,
#                                             min_child_weight = 3,
#                                             subsample = 0.5,
#                                             colsample_bytree = 0.7,
#                                             alpha = 0.1,
#                                             eta = 0.1)
#
#
# # define model evaluation method
# cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
# # evaluate model
# r2_scorer = make_scorer(r2_score)
# SVRscores = cross_val_score(xgbr, X, y, scoring = r2_scorer, cv=cv, n_jobs=-1)
# print('Mean MAE: %.3f (%.3f)' % (mean(SVRscores), std(SVRscores)))
#
# _ = xgbr.fit(X_train, y_train)
# y_pred = xgbr.predict(X_test)
# from sklearn.metrics import r2_score
# score = r2_score(np.array(y_test), y_pred)
# print(f"The score is: {score}")

# ------------- PREDICT ON THE TEST REAL TEST DATA -------------------
xgbr = xgb.XGBRegressor (max_depth = 5,
                                            n_estimators = 100,
                                            min_child_weight = 3,
                                            subsample = 0.8,
                                            alpha = 0.5,
                                            gamma = 15,
                                            colsample_bytree=0.7,
                                            eta = 0.2)
_ = xgbr.fit(X, y)
X = pd.read_csv('X_test.csv', float_precision='high').drop('id', axis=1)
# fill the missing values with the median
X = X.fillna(X.median())
X = X[selectedfeatures]

y_pred = xgbr.predict(X)

f = open("boost.csv", "w")
f.write("id,y\n")

for i in range(len(y_pred)):
    f.write(str(float(i)) + "," + str(y_pred[i]) + "\n")
f.close()
