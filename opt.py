import cvxpy as cp
import numpy as np

n = 10
C = 15

z = 2 * np.random.randint(2, size=n) - 1
y = np.random.randn(n, n)
Q = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        Q[i, j] = z[i] * z[j] * y[i].T @ y[j]

e = np.ones(n)

# Define and solve the CVXPY problem.
alpha = cp.Variable(n)
prob = cp.Problem(
    cp.Minimize(0.5 * cp.quad_form(alpha, Q) - e.T @ alpha), [alpha >= 0, alpha <= C]
)
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(alpha.value)
print("A dual solution is")
print(prob.constraints[0].dual_value)
