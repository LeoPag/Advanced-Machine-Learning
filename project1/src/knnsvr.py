import numpy as np
import csv
import pandas as pd

X_test = pd.read_csv('X_test.csv', float_precision='high').drop('id', axis=1)
X = pd.read_csv('X_train.csv', float_precision='high').drop('id', axis=1)
y = pd.read_csv('y_train.csv', float_precision='high').drop('id', axis=1)
Xbucato = X
Xtestbucato = X_test
X = X.fillna(X.median())
X_test = X_test.fillna(X_test.median())




from sklearn.ensemble import IsolationForest

contamination = 0.01


#Isolation Forest on X, features
model = IsolationForest(contamination = contamination, n_estimators = 1000)
model.fit(X)
anomalies = model.predict(X)

y = y[anomalies == 1]
print(np.shape(y))
X = X[anomalies == 1]
print(np.shape(X))
Xbucato = Xbucato[anomalies == 1]







correlation_threshold = 0.1


Xprova = X
Xprova["y"] = y
#print(Xprova)
correlation = (abs(Xprova.corr()["y"]))
count = 0
for i in correlation:
  if(i > correlation_threshold):
    count = count + 1
    #print(count,i)
print("COUNT IS:", count - 1) #correlation (y,y) is always 1

selectedfeatures = abs(Xprova.corr()["y"][abs(Xprova.corr()["y"]) > correlation_threshold].drop("y")).index.tolist()
print(selectedfeatures)

X = X[selectedfeatures]
Xbucato = Xbucato[selectedfeatures]
X_test = X_test[selectedfeatures]
Xtestbucato = Xtestbucato[selectedfeatures]





from sklearn.preprocessing import MinMaxScaler
# define min max scaler
scaler = MinMaxScaler()

max_X_train = np.amax(np.array(X), 0)
min_X_train = np.amin(np.array(X), 0)
# transform data
X = scaler.fit_transform(X)
Xbucato = scaler.fit_transform(Xbucato)
X_test = np.array(X_test)

for i in range(X_test.shape[1]):
    column = X_test[:, i]
    column_std = (column - min_X_train[i]) / (max_X_train[i] - min_X_train[i])
    X_test[:, i] = column_std

Xtestbucato = np.array(Xtestbucato)
for i in range(Xtestbucato.shape[1]):
    column = Xtestbucato[:, i]
    column_std = (column - min_X_train[i]) / (max_X_train[i] - min_X_train[i])
    Xtestbucato[:, i] = column_std



from sklearn.impute import KNNImputer

n_features_selected = len(X[0,:])
print(Xbucato)
print(np.shape(Xbucato))
#final imputation x_train


#compute autocorrelation
R_train = np.corrcoef(np.transpose(X))
R_test = np.corrcoef(np.transpose(X_test))

#print(R_train)
#print(R_test)

print("step - 1")

#deleate values of low autocorrelation
boolean_corr_train = np.zeros((n_features_selected, n_features_selected))
boolean_corr_test = np.zeros((n_features_selected, n_features_selected))

#print(n_features_selected)
for i in range(n_features_selected):
    for j in range(n_features_selected):
        if abs(R_test[i,j]) >= 0.2:
            boolean_corr_test[i,j]=R_test[i,j]

for i in range(n_features_selected):
    for j in range(n_features_selected):
        if abs(R_train[i,j]) >= 0.2:
            boolean_corr_train[i,j]=R_train[i,j]


#print(boolean_corr_train)
#print(boolean_corr_test)
print("step - 2")

#dictionary of indexes

high_corr_features_train = {}
for i in range(n_features_selected):
    high_corr_features_train[i] = []
    high_corr_features_train[i].append(i)
    for j in range(n_features_selected):
        if(boolean_corr_train[i][j] >= 0.8 and i!=j ):
            high_corr_features_train[i].append(j)
    if len(high_corr_features_train[i]) == 1:
        for j in range(n_features_selected):
            if(boolean_corr_train[i][j] >= 0.7 and i!=j ):
                high_corr_features_train[i].append(j)
    if len(high_corr_features_train[i]) == 1:
        for j in range(n_features_selected):
            if(boolean_corr_train[i][j] >= 0.6 and i!=j ):
                high_corr_features_train[i].append(j)
    if len(high_corr_features_train[i]) == 1:
        for j in range(n_features_selected):
            if(boolean_corr_train[i][j] >= 0.5 and i!=j ):
                high_corr_features_train[i].append(j)

#print(high_corr_features_train)

high_corr_features_test = {}
for i in range(n_features_selected):
    high_corr_features_test[i] = []
    high_corr_features_test[i].append(i)
    for j in range(n_features_selected):
        if(boolean_corr_test[i][j] !=0 and boolean_corr_test[i][j]!=1 ):
            high_corr_features_test[i].append(j)
    if len(high_corr_features_test[i]) == 1:
        for j in range(n_features_selected):
            if(boolean_corr_test[i][j] >= 0.7 and i!=j ):
                high_corr_features_test[i].append(j)
    if len(high_corr_features_test[i]) == 1:
        for j in range(n_features_selected):
            if(boolean_corr_test[i][j] >= 0.6 and i!=j ):
                high_corr_features_test[i].append(j)
    if len(high_corr_features_test[i]) == 1:
        for j in range(n_features_selected):
            if(boolean_corr_test[i][j] >= 0.5 and i!=j ):
                high_corr_features_test[i].append(j)

#filter matrix with Nan for features and layers selected

print("step - 3")

#imputation with k-nn

n_samples_train = len(Xbucato[:,0])
n_samples_test = len(Xtestbucato[:,0])

X_train_si = np.zeros([n_samples_train, n_features_selected])
X_test_si = np.zeros([n_samples_test, n_features_selected])


for i in range(n_features_selected):
    #building the i-matrix with the i-feature and the most correlated
    correlate_i_matrix_train = Xbucato[:,(high_corr_features_train[i])]
    #imputation on that matrix
    imputer = KNNImputer(n_neighbors=3)
    #extracting only the first column as the i-column
    mat_train_imp = imputer.fit_transform(correlate_i_matrix_train)
    X_train_si[:,i] = mat_train_imp[:,0]

for i in range(n_features_selected):
    #building the i-matrix with the i-feature and the most correlated
    correlate_i_matrix_test = Xtestbucato[:,(high_corr_features_test[i])]
    #imputation on that matrix
    imputer = KNNImputer(n_neighbors=3)
    #extracting only the first column as the i-column
    mat_test_imp = imputer.fit_transform(correlate_i_matrix_test)
    X_test_si[:,i] = mat_test_imp[:,0]


#y_train_si = y_train_sfs
"""
print('input is X_train_sfs,X_test_sfs, y_train_sfs')
print('output is X_train_si,X_test_si, y_train_si')
print(X_train_si.shape)
print(X_test_si.shape)
print(y_train_si.shape)
"""

X = X_train_si
X_test = X_test_si

print(X)
print(np.shape(X))









from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import *

kernel = RBF(length_scale=1.25)
regressor = SVR(kernel = kernel, epsilon = 0.01, C=100)

# # define model evaluation method
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# # evaluate model
# r2_scorer = make_scorer(r2_score)
# SVRscores = cross_val_score(regressor, X, y, scoring = r2_scorer, cv=cv, n_jobs=-1)
# print('Mean MAE: %.3f (%.3f)' % (mean(SVRscores), std(SVRscores)))

_ = regressor.fit(X, y)
y_pred = regressor.predict(X_test)
"""
from sklearn.metrics import r2_score
score = r2_score(np.array(y_test), y_pred)
print(f"The score is: {score}")
"""

f = open("output_svr_dio.csv", "w")
f.write("id,y\n")
for i in range(len(y_pred)):
    f.write(str(float(i)) + "," + str(y_pred[i]) + "\n")
f.close()
