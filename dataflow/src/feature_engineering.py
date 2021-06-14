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
