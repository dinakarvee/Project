import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


# -----------------------------
# CREATE LABELS
# -----------------------------
def create_labels(df):

    conditions = [
        (df["group_size"] < 5),
        (df["rarity_score"] < 0.001)
    ]

    choices = [1, 0.5]

    df["risk_label"] = 0  # default low risk
    df.loc[conditions[0], "risk_label"] = 1
    df.loc[conditions[1] & (df["group_size"] >= 5), "risk_label"] = 0.5

    return df


# -----------------------------
# PREPARE FEATURES
# -----------------------------
def prepare_features(df):

    features = [
        "group_size",
        "uniqueness_score",
        "rarity_score",
        "location_granularity",
        "time_granularity",
        "sensitive_count"
    ]

    X = df[features]
    y = df["risk_label"]

    return X, y


# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model(df):

    df = create_labels(df)

    X, y = prepare_features(df)

    # Convert labels to classification
    y = y.replace({0.5: 1})  # simplify to binary classification

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))

    return model


# -----------------------------
# PREDICT RISK
# -----------------------------
def predict_risk(model, df):

    features = [
        "group_size",
        "uniqueness_score",
        "rarity_score",
        "location_granularity",
        "time_granularity",
        "sensitive_count"
    ]

    df["risk_score"] = model.predict_proba(df[features])[:, 1]

    return df


# -----------------------------
# SAVE MODEL
# -----------------------------
def save_model(model, path="models/risk_model.pkl"):
    joblib.dump(model, path)
