import cvxpy as cp
import numpy as np
import pandas as pd

C = 15

# data = pd.read_csv('datasets/breastCancer.csv')
# n = data.shape[0]
# y = data.loc[:, (data.columns != 'diagnosis') & (data.columns != 'id')].to_numpy()
# z = data.loc[:, 'diagnosis'].replace('B', -1).replace('M', 1).to_numpy()

data = pd.read_csv('datasets/diabetes.csv')
n = data.shape[0]
y = data.loc[:, data.columns != 'Outcome'].to_numpy()
z = data.loc[:, 'Outcome'].replace(0, -1).to_numpy()

# z = 2 * np.random.randint(2, size=n) - 1
# y = np.random.rand(n, n) * 100000

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
