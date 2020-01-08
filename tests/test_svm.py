import numpy as np
from optsvm.svm import SVM


def test_linear_separable():
    X = np.array([[3, 4], [1, 4], [2, 3], [6, -1], [7, -1], [5, -3]])
    y = np.array([-1, -1, -1, 1, 1, 1,])

    svm = SVM(C=10)

    w, b = svm.fit(X, y)
    np.testing.assert_allclose(w, np.array([[0.25], [-0.25]]))
    np.testing.assert_allclose(b[0], np.array([-0.75]))


def test_not_linear_separable():
    X = np.array([[3, 4], [1, 4], [2, 3], [6, -1], [7, -1], [5, -3], [2, 4]])
    y = np.array([-1, -1, -1, 1, 1, 1, 1])

    svm = SVM(C=10)

    w, b = svm.fit(X, y)
    np.testing.assert_allclose(w, np.array([[0.25], [-0.25]]))
    np.testing.assert_allclose(b[0], np.array([-0.75]))

