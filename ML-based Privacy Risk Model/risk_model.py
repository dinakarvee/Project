import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


# -----------------------------
# CREATE LABELS
# -----------------------------
def create_labels(df):

    df["risk_label"] = 0

    # High risk rules
    df.loc[
        (df["group_size"] < 5) |
        ((df["group_size"] < 15) & (df["sensitive_count"] >= 1)),
        "risk_label"
    ] = 1

    # Add controlled randomness (10% of remaining)
    remaining_idx = df[df["risk_label"] == 0].sample(frac=0.1, random_state=42).index
    df.loc[remaining_idx, "risk_label"] = 1

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

    # STEP 1: Create labels FIRST
    df = create_labels(df)

    # STEP 2: Now you can check distribution
    print("Label Distribution:\n", df["risk_label"].value_counts())

    # STEP 3: Prepare features
    X, y = prepare_features(df)

    # Convert to binary
    y = (y >= 0.5).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")

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
