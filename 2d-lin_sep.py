import numpy as np
from matplotlib import pyplot as plt
from optsvm.svm import SVM

x_neg = np.array([[3, 4], [1, 4], [2, 3]])
y_neg = np.array([-1, -1, -1])
x_pos = np.array([[6, -1], [7, -1], [5, -3]])
y_pos = np.array([1, 1, 1])
x1 = np.linspace(-10, 10)
x = np.vstack((np.linspace(-10, 10), np.linspace(-10, 10)))

# Data for the next section
X = np.vstack((x_neg, x_pos))
y = np.concatenate((y_neg, y_pos))

# Plot
fig = plt.figure(figsize=(10, 10))
plt.scatter(x_neg[:, 0], x_neg[:, 1], marker="x", color="r", label="Negative -1")
plt.scatter(x_pos[:, 0], x_pos[:, 1], marker="o", color="b", label="Positive +1")
plt.plot(x1, x1 - 3, color="darkblue")
plt.plot(x1, x1 - 7, linestyle="--", alpha=0.3, color="b")
plt.plot(x1, x1 + 1, linestyle="--", alpha=0.3, color="r")
plt.xlim(-2, 12)
plt.ylim(-7, 7)
plt.xticks(np.arange(0, 10, step=1))
plt.yticks(np.arange(-5, 5, step=1))

# Lines
plt.axvline(0, color="black", alpha=0.5)
plt.axhline(0, color="black", alpha=0.5)
plt.plot([2, 6], [3, -1], linestyle="-", color="darkblue", alpha=0.5)
plt.plot([4, 6], [1, 1], [6, 6], [1, -1], linestyle=":", color="darkblue", alpha=0.5)
plt.plot(
    [0, 1.5], [0, -1.5], [6, 6], [1, -1], linestyle=":", color="darkblue", alpha=0.5
)

# Annotations
plt.annotate(s="$A \ (6,-1)$", xy=(5, -1), xytext=(6, -1.5))
plt.annotate(
    s="$B \ (2,3)$", xy=(2, 3), xytext=(2, 3.5)
)  # , arrowprops = {'width':.2, 'headwidth':8})
plt.annotate(s="$2$", xy=(5, 1.2), xytext=(5, 1.2))
plt.annotate(s="$2$", xy=(6.2, 0.5), xytext=(6.2, 0.5))
plt.annotate(s="$2\sqrt{2}$", xy=(4.5, -0.5), xytext=(4.5, -0.5))
plt.annotate(s="$2\sqrt{2}$", xy=(2.5, 1.5), xytext=(2.5, 1.5))
plt.annotate(s="$w^Tx + b = 0$", xy=(8, 4.5), xytext=(8, 4.5))
plt.annotate(
    s="$(\\frac{1}{4},-\\frac{1}{4}) \\binom{x_1}{x_2}- \\frac{3}{4} = 0$",
    xy=(7.5, 4),
    xytext=(7.5, 4),
)
plt.annotate(s="$\\frac{3}{\sqrt{2}}$", xy=(0.5, -1), xytext=(0.5, -1))

# Labels and show
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend(loc="lower right")
plt.show()

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
