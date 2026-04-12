import pandas as pd
from config import K_ANONYMITY, RISK_THRESHOLD_LOW, RISK_THRESHOLD_HIGH


def generalize_location(df):
    df["city"] = "MASKED"
    return df


def generalize_time(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.date
    return df


def generalize_payment(df):
    def bucket(x):
        if x == 0:
            return "0"
        elif x < 10:
            return "1-10"
        elif x < 50:
            return "10-50"
        else:
            return "50+"

    df["payment_amount"] = df["payment_amount"].apply(bucket)
    return df


def suppress_high_risk(df):
    df.loc[df["risk_score"] > RISK_THRESHOLD_HIGH, "device_id"] = "REMOVED"
    df.loc[df["risk_score"] > RISK_THRESHOLD_HIGH, "CVRoot"] = "REMOVED"
    return df


def apply_anonymization(df):

    medium_risk = df[
        (df["risk_score"] >= RISK_THRESHOLD_LOW) &
        (df["risk_score"] <= RISK_THRESHOLD_HIGH)
    ]

    df.loc[medium_risk.index, "city"] = "MASKED"
    df.loc[medium_risk.index, "timestamp"] = pd.to_datetime(
        df.loc[medium_risk.index, "timestamp"]
    ).dt.date

    def bucket(x):
        if x == 0:
            return "0"
        elif x < 10:
            return "1-10"
        elif x < 50:
            return "10-50"
        else:
            return "50+"

    df.loc[medium_risk.index, "payment_amount"] = df.loc[
        medium_risk.index, "payment_amount"
    ].apply(bucket)

    df = suppress_high_risk(df)

    return df


def enforce_k_anonymity(df):

    group_cols = ["city", "app_category", "timestamp"]

    group_sizes = df.groupby(group_cols).size().reset_index(name="group_size_new")

    df = df.merge(group_sizes, on=group_cols, how="left")

    small_groups = df["group_size_new"] < K_ANONYMITY

    df.loc[small_groups, "city"] = "OTHER"
    df.loc[small_groups, "timestamp"] = pd.to_datetime(df["timestamp"]).dt.date

    df.drop(columns=["group_size_new"], inplace=True)

    return df


def anonymization_pipeline(df):

    print("🔹 Applying anonymization...")

    df = apply_anonymization(df)
    df = enforce_k_anonymity(df)

    return df
