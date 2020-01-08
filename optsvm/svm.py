import numpy as np
import cvxpy as cp


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

        w = ((y * alphas).T @ X).reshape(-1, 1)
        S = (alphas > 1e-4).flatten()
        b = y[S] - np.dot(X[S], w)
        self._w = w
        self._b = b

        return w, b

    def predict(self, X):
        # sign( x·w+b )
        # TODO check b[0] correctness
        dot_result = np.sign(np.dot(np.array(X), self._w) + self._b[0])
        return dot_result.astype(int).flatten()
