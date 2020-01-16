import numpy as np
import cvxpy as cp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder


def b_average(w, sv, sv_y):
    return np.sum(sv_y - np.dot(sv, w)) / len(sv)


def b_nearest_sv(w, sv, sv_y):
    dots = np.dot(sv, w)

    neg = dots[sv_y == -1]
    pos = dots[sv_y == 1]
    if len(neg) == 0 and len(pos) == 0:
        return 0
    elif len(neg) == 0:
        return -1 * min(pos)
    elif len(pos) == 0:
        return -1 * max(neg)
    else:
        return -1 * (max(neg) + min(pos)) / 2


class SVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=10):
        self.C = C

    def _more_tags(self):
        return {"requires_fit": True, "binary_only": True}

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        m, n = X.shape
        y = y.reshape(-1, 1) * 1.0
        X_dash = y * X
        H = np.dot(X_dash, X_dash.T) * 1.0

        P = H
        q = -np.ones((m, 1))
        G = np.vstack((np.eye(m) * -1, np.eye(m)))
        h = np.hstack((np.zeros(m), np.ones(m) * self.C))
        A = y.reshape(1, -1)
        b = np.zeros(1)

        alpha = cp.Variable(m)
        prob = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(alpha, P) + q.T @ alpha),
            [G @ alpha <= h, A @ alpha == b],
        )
        prob.solve()

        alphas = alpha.value.reshape(-1, 1)
        applicable_lagrangian = (self.C > alphas).flatten() & (alphas > 1e-4).flatten()

        w = ((y * alphas).T @ X).reshape(-1, 1)

        support_vectors = X[applicable_lagrangian]
        support_vectors_y = y[applicable_lagrangian]

        b = b_nearest_sv(w, support_vectors, support_vectors_y)
        self.w_ = w
        self.b_ = b

        return self

    def predict(self, X):
        # check_is_fitted(self)
        X = check_array(X)

        # sign( x·w+b )
        dot_result = np.sign(np.dot(np.array(X), self.w_) + self.b_)
        return dot_result.astype(int).flatten()
