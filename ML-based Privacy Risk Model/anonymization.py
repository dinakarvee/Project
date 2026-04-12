import pandas as pd

# -----------------------------
# GENERALIZATION FUNCTIONS
# -----------------------------
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


# -----------------------------
# SUPPRESSION
# -----------------------------
def suppress_high_risk(df):
    df.loc[df["risk_score"] > 0.7, "device_id"] = "REMOVED"
    df.loc[df["risk_score"] > 0.7, "CVRoot"] = "REMOVED"
    return df


# -----------------------------
# APPLY ANONYMIZATION
# -----------------------------
def apply_anonymization(df):

    # Medium risk → generalization
    medium_risk = df[(df["risk_score"] >= 0.3) & (df["risk_score"] <= 0.7)]

    df.loc[medium_risk.index] = generalize_location(df.loc[medium_risk.index])
    df.loc[medium_risk.index] = generalize_time(df.loc[medium_risk.index])
    df.loc[medium_risk.index] = generalize_payment(df.loc[medium_risk.index])

    # High risk → suppression
    df = suppress_high_risk(df)

    return df


# -----------------------------
# K-ANONYMITY ENFORCEMENT
# -----------------------------
def enforce_k_anonymity(df, k=5):

    group_cols = ["city", "app_category", "timestamp"]

    group_sizes = df.groupby(group_cols).size().reset_index(name="group_size")

    df = df.merge(group_sizes, on=group_cols, how="left", suffixes=("", "_new"))

    # Suppress groups smaller than k
    small_groups = df["group_size_new"] < k

    df.loc[small_groups, "city"] = "OTHER"
    df.loc[small_groups, "timestamp"] = pd.to_datetime(df["timestamp"]).dt.date

    df.drop(columns=["group_size_new"], inplace=True)

    return df


# -----------------------------
# FULL PIPELINE
# -----------------------------
def anonymization_pipeline(df):

    print("Applying anonymization...")

    df = apply_anonymization(df)

    df = enforce_k_anonymity(df, k=5)

    return df
