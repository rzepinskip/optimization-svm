import pandas as pd

def preprocess_diabetes(data, className):
    X = data.loc[:, data.columns != className]
    y = data.loc[:, className].replace(0, -1)
    return X, y

def preprocess_adult(data, className):
    X = pd.get_dummies(data.loc[:, data.columns != className])
    y = data.loc[:, className].replace("<=50K", -1).replace(">50K", 1)
    return X, y

def preprocess_occupancy(data, className):
    X = data.loc[:, (data.columns != className) & (data.columns != "date")]
    y = data.loc[:, className].replace(0, -1)
    return X, y
