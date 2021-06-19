# https://codelabs.developers.google.com/codelabs/vertex-ai-custom-code-training#3

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle

"""
from src.feature_engineering import (
    dummy_dict,
    internet_dict,
    train_yesno_cols,
    internet_cols,
    data_transformations,
)
"""

dummy_dict = {"Yes": 1, "No": 0}
internet_dict = {"No": 0, "No internet service": 1, "Yes": 2}
train_yesno_cols = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]
inference_yesno_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
internet_cols = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# preprocessing categorical features
def data_transformations(data, dummy_dict, internet_dict, yesno_cols, internet_cols):

    data[yesno_cols] = data[yesno_cols].apply(lambda x: x.map(dummy_dict))
    data[internet_cols] = data[internet_cols].apply(lambda x: x.map(internet_dict))

    # manual map
    data["gender"] = data["gender"].map({"Female": 0, "Male": 1})
    data["MultipleLines"] = data["MultipleLines"].map(
        {"No": 0, "No phone service": 1, "Yes": 2}
    )
    data["InternetService"] = data["InternetService"].map(
        {"DSL": 0, "Fiber optic": 1, "No": 2}
    )
    data["Contract"] = data["Contract"].map(
        {"Month-to-month": 0, "One year": 1, "Two year": 2}
    )
    data["PaymentMethod"] = data["PaymentMethod"].map(
        {
            "Bank transfer (automatic)": 0,
            "Credit card (automatic)": 1,
            "Electronic check": 2,
            "Mailed check": 3,
        }
    )
    return data


data = pd.read_csv("gs://prav_timeseries_features/data/trainingSet.csv")
data.head()
data.shape
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data.isnull().sum()

data.dropna(inplace=True)

data.shape

data = data_transformations(
    data, dummy_dict, internet_dict, train_yesno_cols, internet_cols
)

data.head()

# modeling
y = data["Churn"].values
X = data.drop(columns=["customerID", "Churn"])

# aim is not to build world class model, rather a simple model for pipeline build/testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)
model_rf = RandomForestClassifier(
    n_estimators=1000,
    oob_score=True,
    n_jobs=-1,
    random_state=50,
    max_features="auto",
    max_leaf_nodes=30,
)
model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)

# cross validation scoring
print(roc_auc_score(y_test, prediction_test))
"""
# store model object for inference
with open("./models/model_rf.pickle", "wb") as handle:
    pickle.dump(model_rf, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
