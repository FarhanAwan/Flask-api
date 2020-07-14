# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
print(X)
print(y)
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y=y.reshape(-1, 1)
onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y).toarray()
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
cm = confusion_matrix(y_test.argmax(axis=1),y_pred.argmax(axis=1))

import pickle
import joblib
filename = 'expert_model.pkl'
joblib.dump(classifier, filename)

#model = joblib.load('expert_model')

#sTesting a single observation
testd = pd.read_csv('testdata.csv')
testd = testd.values
print(testd)
labelencoder_test = LabelEncoder()
testd[:, 0] = labelencoder_test.fit_transform(testd[:, 0])
y_show = classifier.predict(testd)
print(y_show)





























# print(y_show)
# result = ""
# if int(y_show[0][0]) == 1:
#     result = "Faculty OF  Engineering"
#     print("Faculty OF  Engineering")
# elif int(y_show[0][1]) == 1:
#     print("Faculty Of Arts and Social Sciences")
#     result = "Faculty Of Arts and Social Sciences"
# elif int(y_show[0][2]) == 1:
#     print("Faculty Of Pharmacy and Pharmaceutical Sciences")
#     result = "Faculty Of Pharmacy and Pharmaceutical Sciences"
# elif int(y_show[0][3]) == 1:
#     print("Faculty Of Science")
#     result = "Faculty Of Science"
    
# f = open("result.txt", "w")
# f.write(result)



    
    
    


