import cvxpy as cp
import numpy as np
import pandas as pd


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

# Importing with custom names to avoid issues with numpy / sympy matrix
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

C = 10
m, n = X.shape
y = y.reshape(-1, 1) * 1.0
X_dash = y * X
H = np.dot(X_dash, X_dash.T) * 1.0

# Converting into cvxopt format
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

# Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol["x"])

w = ((y * alphas).T @ X).reshape(-1, 1)
S = (alphas > 1e-4).flatten()
b = y[S] - np.dot(X[S], w)

# Display results
print("---Our results")
print("w = ", w.flatten())
print("b = ", b[0])

from sklearn.svm import SVC

clf = SVC(C=10, kernel="linear")
clf.fit(X, y.ravel())

print("---SVM library")
print("w = ", clf.coef_)
print("b = ", clf.intercept_)

