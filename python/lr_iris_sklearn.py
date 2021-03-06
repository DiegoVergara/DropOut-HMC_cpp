import numpy as np
import pandas as pd
from sklearn import linear_model
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

X_train = pd.read_csv("../data/IRIS/X_train.csv", sep =",", names = None, header = None)
Y_train = pd.read_csv("../data/IRIS/Y_train.csv", sep =",", names = None, header = None)
X_test = pd.read_csv("../data/IRIS/X_test.csv", sep =",", names = None, header = None)
Y_test = pd.read_csv("../data/IRIS/Y_test.csv", sep =",", names = None, header = None)

print(X_train.shape)
print(X_test.shape)
#logreg = linear_model.SGDClassifier(loss='log',penalty='l2',alpha=0.1,learning_rate='constant', eta0 = 0.01, fit_intercept=True,n_iter=100,verbose=1)
logreg = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=1, warm_start=False, n_jobs=1)
logreg.fit(X_train,Y_train)
Y_pred=logreg.predict(X_test)
print(classification_report(Y_test, Y_pred))
print confusion_matrix(Y_test, Y_pred)