import cvxpy as cp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

x_neg = np.array([[3, 4], [1, 4], [2, 3]])
y_neg = np.array([-1, -1, -1])
x_pos = np.array([[6, -1], [7, -1], [5, -3]])
y_pos = np.array([1, 1, 1])
x1 = np.linspace(-10, 10)
x = np.vstack((np.linspace(-10, 10), np.linspace(-10, 10)))

# Data for the next section
X = np.vstack((x_pos, x_neg))
y = np.concatenate((y_pos, y_neg))

# #Parameters guessed by inspection
# w = np.array([1,-1]).reshape(-1,1)
# b = -3

# #Plot
# fig = plt.figure(figsize = (10,10))
# plt.scatter(x_neg[:,0], x_neg[:,1], marker = 'x', color = 'r', label = 'Negative -1')
# plt.scatter(x_pos[:,0], x_pos[:,1], marker = 'o', color = 'b',label = 'Positive +1')
# plt.plot(x1, x1  - 3, color = 'darkblue')
# plt.plot(x1, x1  - 7, linestyle = '--', alpha = .3, color = 'b')
# plt.plot(x1, x1  + 1, linestyle = '--', alpha = .3, color = 'r')
# plt.xlim(-2,12)
# plt.ylim(-7,7)
# plt.xticks(np.arange(0, 10, step=1))
# plt.yticks(np.arange(-5, 5, step=1))

# #Lines
# plt.axvline(0, color = 'black', alpha = .5)
# plt.axhline(0,color = 'black', alpha = .5)
# plt.plot([2,6],[3,-1], linestyle = '-', color = 'darkblue', alpha = .5 )
# plt.plot([4,6],[1,1],[6,6],[1,-1], linestyle = ':', color = 'darkblue', alpha = .5 )
# plt.plot([0,1.5],[0,-1.5],[6,6],[1,-1], linestyle = ':', color = 'darkblue', alpha = .5 )

# #Annotations
# plt.annotate(s = '$A \ (6,-1)$', xy = (5,-1), xytext = (6,-1.5))
# plt.annotate(s = '$B \ (2,3)$', xy = (2,3), xytext = (2,3.5))#, arrowprops = {'width':.2, 'headwidth':8})
# plt.annotate(s = '$2$', xy = (5,1.2), xytext = (5,1.2) )
# plt.annotate(s = '$2$', xy = (6.2,.5), xytext = (6.2,.5))
# plt.annotate(s = '$2\sqrt{2}$', xy = (4.5,-.5), xytext = (4.5,-.5))
# plt.annotate(s = '$2\sqrt{2}$', xy = (2.5,1.5), xytext = (2.5,1.5))
# plt.annotate(s = '$w^Tx + b = 0$', xy = (8,4.5), xytext = (8,4.5))
# plt.annotate(s = '$(\\frac{1}{4},-\\frac{1}{4}) \\binom{x_1}{x_2}- \\frac{3}{4} = 0$', xy = (7.5,4), xytext = (7.5,4))
# plt.annotate(s = '$\\frac{3}{\sqrt{2}}$', xy = (.5,-1), xytext = (.5,-1))

# #Labels and show
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# plt.legend(loc = 'lower right')
# plt.show()


m, n = X.shape
y = y.reshape(-1, 1) * 1.0
X_dash = y * X
H = np.dot(X_dash, X_dash.T) * 1.0

# Converting into quadratic program notation
P = H
q = -np.ones((m, 1))
G = -np.eye(m)
h = np.zeros(m)
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

# Selecting the set of indices S corresponding to non zero parameters
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
