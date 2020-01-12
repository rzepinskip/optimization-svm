import numpy as np
import pandas as pd
from optsvm.svm import SVM
from time import process_time
from preprocessing import preprocess_diabetes, preprocess_adult, preprocess_occupancy
from sklearn.metrics import classification_report

datasets = {
    "diabetes": {
        "file": "diabetes.csv",
        "className": "Outcome",
        "preprocess": preprocess_diabetes
    },
    "adult": {
        "file": "adult.data",
        "className": "Outcome",
        "preprocess": preprocess_adult
    },
    "occupancy": {
        "file": "occupancy.txt",
        "className": "Occupancy",
        "preprocess": preprocess_occupancy
    }
}


def read_data(dataset):
    data = pd.read_csv("datasets/" + dataset["file"])
    numeric_columns = list(data.select_dtypes(include=[np.number]).columns.values)
    for column in numeric_columns:
        if column != dataset["className"]:
            data[column] = (data[column] - data[column].min()) / (
                data[column].max() - data[column].min()
            )

    return data

def get_mask_data(X, y, msk):
    return X[msk].to_numpy(), y[msk].to_numpy()


dataset = datasets["diabetes"]

data = read_data(dataset)
X, y = dataset["preprocess"](data, dataset["className"])

msk = np.random.rand(len(data)) < 0.8
X_train, y_train = get_mask_data(X, y, msk)
X_test, y_test = get_mask_data(X, y, ~msk)
n = X_train.shape[0]

svm = SVM(C=10)

start_time = process_time()
w, b = svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
stop_time = process_time()

# Display results
print("---Our results")
print("w = ", w.flatten())
print("b = ", b)
print("time = ", stop_time - start_time)
print(classification_report(y_test, y_pred, labels=[-1, 1]))

from sklearn.svm import SVC

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
