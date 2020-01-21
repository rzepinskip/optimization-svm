import numpy as np
import pandas as pd
from optsvm.svm import SVM
from time import process_time
from preprocessing import preprocess_diabetes, preprocess_adult, preprocess_occupancy
from sklearn.metrics import classification_report
import logging
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os
from scipy.optimize import curve_fit


datasets = {
    "diabetes": {
        "file": "diabetes.csv",
        "className": "Outcome",
        "preprocess": preprocess_diabetes,
    },
    "occupancy": {
        "file": "occupancy.txt",
        "className": "Occupancy",
        "preprocess": preprocess_occupancy,
    },
    "adult": {
        "file": "adult.data",
        "className": "Outcome",
        "preprocess": preprocess_adult,
    },
}

solvers = ["ECOS", "SCS", "OSQP"]


def read_data(dataset):
    data = pd.read_csv("datasets/" + dataset["file"])
    numeric_columns = list(data.select_dtypes(include=[np.number]).columns.values)
    for column in numeric_columns:
        if column != dataset["className"]:
            data[column] = (data[column] - data[column].min()) / (
                data[column].max() - data[column].min()
            )

    return data

def fitting_func(x, a, b):
    return a * x ** b


dataset = datasets["adult"]
data = read_data(dataset)

elapsed_times = {k:[] for k in solvers + ['sklearn']}

step = 5000
sizes = np.concatenate([np.arange(step, len(data), step), np.array([len(data)])])
for size in sizes:
    print("size = ", size)
    sampled_data = data.sample(n=size)
    X, y = dataset["preprocess"](sampled_data, dataset["className"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for solver in solvers:
        if size > 25000 and solver == 'OSQP':
            continue
        try:
            print("solver = ", solver)
            svm = SVM(C=10, solver=solver)

            start_time = process_time()
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            stop_time = process_time()

            t = stop_time - start_time
            elapsed_times[solver].append([size, t])

            # Display results
            print("---Our results")
            print("w = ", svm.w_.flatten())
            print("b = ", svm.b_)
            print("time = ", t)
            print(classification_report(y_test, y_pred, labels=[-1, 1]))
        except: # ValueError as error:
            logging.exception("Too large problem for solver")

    clf = SVC(C=10, kernel="linear")
    start_time = process_time()
    clf.fit(X_train, y_train.ravel())
    y_pred = clf.predict(X_test)
    stop_time = process_time()

    t = stop_time - start_time
    elapsed_times['sklearn'].append([size, t])

    print("---SVM library")
    print("w = ", clf.coef_)
    print("b = ", clf.intercept_)
    print("time = ", t)
    print(classification_report(y_test, y_pred, labels=[-1, 1]))

    directory = 'results/size' + str(size)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for key in elapsed_times.keys():
        d = np.array(elapsed_times[key])
        np.savetxt(directory + "/solver_" + key + ".csv", d)

fig = plt.figure()
for key in elapsed_times.keys():
    d = np.genfromtxt(directory + "/solver_" + key + ".csv")
    if key == 'ECOS':
        f, _ =  curve_fit(fitting_func, d[:, 0], d[:, 1])
        plt.plot(d[:, 0],fitting_func(d[:, 0], *f), linewidth=7, alpha=0.5, label=str(round(f[0], 11))+"*x^" + str(round(f[1], 2)))
    plt.plot(d[:, 0], d[:, 1], label=key)
plt.xlabel("Size of dataset")
plt.ylabel("Time [s]")
plt.legend()
plt.savefig(directory + "/plot.pdf")
