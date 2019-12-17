import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC

dataset = pd.read_csv("datasets/diabetes.csv")

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


classifier = SVC(kernel="linear", random_state=0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

confusion_matrix(Y_test, Y_pred)

accuracy_score(Y_test, Y_pred)
