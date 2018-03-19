import pandas as pandas

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn import metrics

import matplotlib.pyplot as plt

input_file = "datasets/shopping.v2.1.input.csv"
dfInput = pandas.read_csv(input_file, header=0)
output_file = "datasets/shopping.v2.0.output.csv"
dfOutput = pandas.read_csv(output_file, header=0)

X = dfInput.iloc[:, 4:]
y = dfOutput.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

modelLR = LogisticRegression()
modelLR.fit(X_train, y_train)
predictionsLR = modelLR.predict(X_test)

modelSVMLinear = SVC(kernel="linear")
modelSVMLinear.fit(X_train, y_train)
predictionsSVMLinear = modelSVMLinear.predict(X_test)

modelSVMRbf = SVC(kernel="rbf")
modelSVMRbf.fit(X_train, y_train)
predictionsSVMRbf = modelSVMRbf.predict(X_test)

modelSVMSigmoid = SVC(kernel="sigmoid")
modelSVMSigmoid.fit(X_train, y_train)
predictionsSVMSigmoid = modelSVMSigmoid.predict(X_test)

modelSVMPoly = SVC(kernel="poly")
modelSVMPoly.fit(X_train, y_train)
predictionsSVMPoly = modelSVMPoly.predict(X_test)

modelNeighbors = neighbors.KNeighborsClassifier(n_neighbors=10)
modelNeighbors.fit(X_train, y_train)
predictionsNeighbors = modelNeighbors.predict(X_test)

modelRandomForest = RandomForestClassifier(n_estimators=10)
modelRandomForest.fit(X_train, y_train)
predictionsRandomForest = modelRandomForest.predict(X_test)


print("Logistic Regression :", metrics.accuracy_score(y_test, predictionsLR))
print("SVM linear :", metrics.accuracy_score(y_test, predictionsSVMLinear))
print("SVM rbf :", metrics.accuracy_score(y_test, predictionsSVMRbf))
print("SVM sigmoid :", metrics.accuracy_score(y_test, predictionsSVMSigmoid))
print("SVM poly :", metrics.accuracy_score(y_test, predictionsSVMPoly))
print("Neighbors :", metrics.accuracy_score(y_test, predictionsNeighbors))
print("Random Forest :", metrics.accuracy_score(
    y_test, predictionsRandomForest))
