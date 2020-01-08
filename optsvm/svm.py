import numpy as np
import cvxpy as cp


def b_average(w, sv, sv_y):
    return np.sum(sv_y - np.dot(sv, w)) / len(sv)


def b_nearest_sv(w, sv, sv_y):
    dots = np.dot(sv, w)
    return -1 * (max(dots[sv_y == -1]) + min(dots[sv_y == 1])) / 2


class SVM:
    def __init__(self, C=10):
        self._C = C

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        C = self._C
        m, n = X.shape
        y = y.reshape(-1, 1) * 1.0
        X_dash = y * X
        H = np.dot(X_dash, X_dash.T) * 1.0

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
        applicable_lagrangian = (self._C > alphas).flatten() & (alphas > 1e-4).flatten()

        w = ((y * alphas).T @ X).reshape(-1, 1)

        support_vectors = X[applicable_lagrangian]
        support_vectors_y = y[applicable_lagrangian]

        b = b_nearest_sv(w, support_vectors, support_vectors_y)
        self._w = w
        self._b = b

        return w, b

    def predict(self, X):
        # sign( x·w+b )
        dot_result = np.sign(np.dot(np.array(X), self._w) + self._b)
        return dot_result.astype(int).flatten()
