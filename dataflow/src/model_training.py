import numpy as np
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle

from src.feature_engineering import (
    dummy_dict,
    internet_dict,
    train_yesno_cols,
    internet_cols,
    data_transformations,
)

data = pd.read_csv("./data/trainingSet.csv")
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

# store model object for inference
with open("./models/model_rf.pickle", "wb") as handle:
    pickle.dump(model_rf, handle, protocol=pickle.HIGHEST_PROTOCOL)
