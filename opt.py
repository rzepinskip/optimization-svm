import numpy as np
import pandas as pd
from optsvm.svm import SVM
from time import process_time
from preprocessing import preprocess_diabetes, preprocess_adult, preprocess_occupancy
from sklearn.metrics import classification_report
import logging
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

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


for key, dataset in datasets.items():
    print(f"================== {key} ==================\n")

    data = read_data(dataset)
    X, y = dataset["preprocess"](data, dataset["className"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    for solver in solvers:
        try:
            print("solver = ", solver)
            svm = SVM(C=10)

            start_time = process_time()
            w, b = svm.fit(X_train, y_train, solver)
            y_pred = svm.predict(X_test)
            stop_time = process_time()

            # Display results
            print("---Our results")
            print("w = ", w.flatten())
            print("b = ", b)
            print("time = ", stop_time - start_time)
            print(classification_report(y_test, y_pred, labels=[-1, 1]))
        except MemoryError as error:
            logging.exception("Too large problem for solver")

    clf = SVC(C=10, kernel="linear")
    start_time = process_time()
    clf.fit(X_train, y_train.ravel())
    y_pred = clf.predict(X_test)
    stop_time = process_time()

    print("---SVM library")
    print("w = ", clf.coef_)
    print("b = ", clf.intercept_)
    print("time = ", stop_time - start_time)
    print(classification_report(y_test, y_pred, labels=[-1, 1]))
