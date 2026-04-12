import pandas as pd

# -----------------------------
# GROUP SIZE
# -----------------------------
def compute_group_size(df):
    group_cols = ["city", "app_category", "timestamp"]

    group_sizes = df.groupby(group_cols).size().reset_index(name="group_size")

    df = df.merge(group_sizes, on=group_cols, how="left")

    return df

# -----------------------------
# UNIQUENESS
# -----------------------------
def compute_uniqueness(df):
    df["uniqueness_score"] = 1 / df["group_size"]
    return df


# -----------------------------
# RARITY
# -----------------------------
def compute_rarity(df):
    total = len(df)
    df["rarity_score"] = df["group_size"] / total
    return df


# -----------------------------
# LOCATION GRANULARITY
# -----------------------------
def compute_location_granularity(df):
    # since we are using city-level data
    df["location_granularity"] = 1.0
    return df


# -----------------------------
# TIME GRANULARITY
# -----------------------------
def compute_time_granularity(df):
    df["time_granularity"] = 1.0  # full timestamp
    return df


# -----------------------------
# SENSITIVE COUNT
# -----------------------------
def compute_sensitive(df):
    df["is_sensitive_app"] = df["app_category"].isin(["health", "finance"]).astype(int)
    df["has_payment"] = (df["payment_amount"] > 0).astype(int)

    df["sensitive_count"] = df["is_sensitive_app"] + df["has_payment"]

    return df


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def feature_engineering_pipeline(df):

    df = compute_group_size(df)
    df = compute_uniqueness(df)
    df = compute_rarity(df)
    df = compute_location_granularity(df)
    df = compute_time_granularity(df)
    df = compute_sensitive(df)

    return df