from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

'''
trains and tests SVM classifiers and produces metrics (accuracy, precision, and recall)
'''

data = 'datasmall.csv'

cell_df = pd.read_csv(data)

feature_df = cell_df[['1', '2', '3', '4', '5', '6']]

X = np.asarray(feature_df)

y = np.asarray(cell_df['Class'])

state = 10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=state) # 70% training and 30% test

clf = svm.SVC(kernel='rbf') # RBF Kernel

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Metrics
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))