import numpy as np
from matplotlib import pyplot as plt
from optsvm.svm import SVM

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

X = np.vstack((x_pos, x_neg))
y = np.concatenate((y_pos, y_neg))
svm = SVM(C=10)

w, b = svm.fit(X, y)

print("---Our results")
print("w = ", w.flatten())
print("b = ", b[0])

from sklearn.svm import SVC

clf = SVC(C=10, kernel="linear")
clf.fit(X, y.ravel())

print("---SVM library")
print("w = ", clf.coef_)
print("b = ", clf.intercept_)
