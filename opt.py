import numpy as np
import pandas as pd
from optsvm.svm import SVM

className = "Outcome"


def read_data(file):
    data = pd.read_csv(file)
    for column in data:
        if column != className:
            data[column] = (data[column] - data[column].min()) / (
                data[column].max() - data[column].min()
            )

    return data


def get_vars_and_classes(data):
    y = data.loc[:, data.columns != className].to_numpy()
    z = data.loc[:, className].replace(0, -1).to_numpy()
    return y, z


data = read_data("datasets/diabetes.csv")
msk = np.random.rand(len(data)) < 0.8
y, z = get_vars_and_classes(data[msk])
y_test, z_test = get_vars_and_classes(data[~msk])
n = y.shape[0]

X = y
y = z

svm = SVM(C=10)

w, b = svm.fit(X, y)

# Display results
print("---Our results")
print("w = ", w.flatten())
print("b = ", b)

from sklearn.svm import SVC

clf = SVC(C=10, kernel="linear")
clf.fit(X, y.ravel())

print("---SVM library")
print("w = ", clf.coef_)
print("b = ", clf.intercept_)
