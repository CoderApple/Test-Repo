#for Mushroom Dataset
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('mushroom_data.csv')
df.head()
df.shape
le = LabelEncoder()
ds = df.apply(func=le.fit_transform)
ds.head()

data = ds.values # becomes a list
data

X = data[:, 1:] # all colsexcept 1st
y = data[:, 0] # all rows, just 0 col
X.shape, y.shape


# X = df.drop('class', axis=1)
# y = df['class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
classifier = GaussianNB()

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print('Accuracy:', accuracy)
