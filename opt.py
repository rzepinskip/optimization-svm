import cvxpy as cp
import numpy as np
import pandas as pd


C = 15
className = 'Outcome'


def read_data(file):
    data = pd.read_csv(file)
    for column in data:
        if column != className:
            data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())

    return data

def get_vars_and_classes(data):
    y = data.loc[:, data.columns != className].to_numpy()
    z = data.loc[:, className].replace(0, -1).to_numpy()
    return y, z


data = read_data('datasets/diabetes.csv')
msk = np.random.rand(len(data)) < 0.8
y, z = get_vars_and_classes(data[msk])
y_test, z_test = get_vars_and_classes(data[~msk])
n = y.shape[0]

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

print("\nThe optimal value is", prob.value)
print("A solution x is")
print(alpha.value)
print("A dual solution is")
print(prob.constraints[0].dual_value)

# we should add calculated h() function for accordingly train and test data 
# using alpha.value or prob.constraints[0].dual_value or something else (?)

# train_error = (np.sign(alpha.value @ y) != np.sign(z @ y)).sum() / n
# print(train_error)
# test_error = (np.sign(alpha.value @ y_test) != np.sign(z_test @ y_test)).sum() / y_test.shape[0]
# print(test_error)