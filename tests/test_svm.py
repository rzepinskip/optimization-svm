import numpy as np
from optsvm.svm import SVM


def test_linear_separable():
    X = np.array([[3, 4], [1, 4], [2, 3], [6, -1], [7, -1], [5, -3]])
    y = np.array([-1, -1, -1, 1, 1, 1,])

    svm = SVM(C=10)

    svm.fit(X, y)
    assert np.allclose(svm.w_, np.array([[0.25], [-0.25]]))
    assert np.allclose(svm.b_, np.array([-0.75]))

    X_test = [[2, 4], [6, -2]]
    y_test = [-1, 1]
    assert np.array_equal(svm.predict(X_test), y_test)


def test_not_linear_separable():
    X = np.array([[3, 4], [1, 4], [2, 3], [6, -1], [7, -1], [5, -3], [2, 4]])
    y = np.array([-1, -1, -1, 1, 1, 1, 1])

    svm = SVM(C=10)

    svm.fit(X, y)
    assert np.allclose(svm.w_, np.array([[0.25], [-0.25]]))
    assert np.allclose(svm.b_, np.array([-0.75]))

    X_test = [[2, 4], [6, -2]]
    y_test = [-1, 1]
    assert np.array_equal(svm.predict(X_test), y_test)
