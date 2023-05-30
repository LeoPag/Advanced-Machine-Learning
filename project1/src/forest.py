import numpy as np
import csv
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(6776)

# -------------------------  GET THE DATA --------------------------------
X = pd.read_csv('X_train.csv', float_precision='high').drop('id', axis=1)
y = pd.read_csv('y_train.csv', float_precision='high').drop('id', axis=1)
# fill the missing values with the median
X = X.fillna(X.median())

# -------------------------  GET THE OUTLIERS  --------------------------------
model = IsolationForest(contamination = 0.11, n_estimators = 1000)
model.fit(X)
anomalies = model.predict(X)
y = y[anomalies == 1]
X = X[anomalies == 1]

# --------------- SPLIT IN TRAIN, TEST AND VALIDATION ---------------------
# split X in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

# -------------------------  INPUT REDUCTION  --------------------------------
correlation_threshold = 0.2

Xprova = X_train
Xprova["y"] = y_train
correlation = (abs(Xprova.corr()["y"]))

selectedfeatures = abs(Xprova.corr()["y"][abs(Xprova.corr()["y"]) > correlation_threshold].drop("y")).index.tolist()
X = X[selectedfeatures]
X_train = X_train[selectedfeatures]
X_test = X_test[selectedfeatures]

from sklearn.preprocessing import MinMaxScaler

# define min max scaler
scaler = MinMaxScaler()

# --------------- CREATE THE MODEL ---------------------
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

forest = RandomForestRegressor(n_estimators=180)
_ = forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)

from sklearn.metrics import r2_score
score = r2_score(np.array(y_test), y_pred)
print(f"The score is: {score}")

# ------------- PREDICT ON THE TEST REAL TEST DATA -------------------
forest = RandomForestRegressor(n_estimators=180)
_ = forest.fit(X, y)
X = pd.read_csv('X_test.csv', float_precision='high').drop('id', axis=1)
# fill the missing values with the median
X = X.fillna(X.median())
X = X[selectedfeatures]
y_pred = forest.predict(X)

f = open("output_forest.csv", "w")
f.write("id,y\n")

for i in range(len(y_pred)):
    f.write(str(float(i)) + "," + str(y_pred[i]) + "\n")
f.close()
