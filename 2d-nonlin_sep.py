import cvxpy as cp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

x_neg = np.array([[3, 4], [1, 4], [2, 3]])
y_neg = np.array([-1, -1, -1])
x_pos = np.array([[6, -1], [7, -1], [5, -3], [2, 4]])
y_pos = np.array([1, 1, 1, 1])
x1 = np.linspace(-10, 10)
x = np.vstack((np.linspace(-10, 10), np.linspace(-10, 10)))


fig = plt.figure(figsize=(10, 10))
plt.scatter(x_neg[:, 0], x_neg[:, 1], marker="x", color="r", label="Negative -1")
plt.scatter(x_pos[:, 0], x_pos[:, 1], marker="o", color="b", label="Positive +1")
plt.plot(x1, x1 - 3, color="darkblue", alpha=0.6, label="Previous boundary")
plt.xlim(0, 10)
plt.ylim(-5, 5)
plt.xticks(np.arange(0, 10, step=1))
plt.yticks(np.arange(-5, 5, step=1))

# Lines
plt.axvline(0, color="black", alpha=0.5)
plt.axhline(0, color="black", alpha=0.5)


plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.legend(loc="lower right")
plt.show()

# New dataset (for later)
X = np.array([[3, 4], [1, 4], [2, 3], [6, -1], [7, -1], [5, -3], [2, 4]])
y = np.array([-1, -1, -1, 1, 1, 1, 1])

# Initializing values and computing H. Note the 1. to force to float type
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

print("---Our results")
print("w = ", w.flatten())
print("b = ", b[0])

from sklearn.svm import SVC

clf = SVC(C=10, kernel="linear")
clf.fit(X, y.ravel())

print("---SVM library")
print("w = ", clf.coef_)
print("b = ", clf.intercept_)
