import cvxpy as cp
import numpy as np
import pandas as pd


def read_data(file, className):
    data = pd.read_csv(file)
    for column in data:
        if column != className:
            data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())

    return data, data.shape[0]


C = 15

# className = 'diagnosis'
# data, n = read_data('datasets/breastCancer.csv', className)
# y = data.loc[:, (data.columns != className) & (data.columns != 'id')].to_numpy()
# z = data.loc[:, className].replace('B', -1).replace('M', 1).to_numpy()

className = 'Outcome'
data, n = read_data('datasets/diabetes.csv', className)
y = data.loc[:, data.columns != className].to_numpy()
z = data.loc[:, className].replace(0, -1).to_numpy()

Q = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        Q[i, j] = z[i] * z[j] * y[i].T @ y[j]

e = np.ones(n)

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
