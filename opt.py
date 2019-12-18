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

C = 10
m, n = X.shape
y = y.reshape(-1, 1) * 1.0
X_dash = y * X
H = np.dot(X_dash, X_dash.T) * 1.0

# Converting into cvxopt format - as previously
P = H
q = -np.ones((m, 1))
G = np.vstack((np.eye(m) * -1, np.eye(m)))
h = np.hstack((np.zeros(m), np.ones(m) * C))
A = y.reshape(1, -1)
b = np.zeros(1)

alpha = cp.Variable(m)
prob = cp.Problem(
    cp.Minimize(0.5 * cp.quad_form(alpha, P) + q.T @ alpha),
    [G @ alpha <= h, A @ alpha == b],
)
prob.solve()

alphas = alpha.value.reshape(-1, 1)

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

