import pandas as pd

from feature_engineering import feature_engineering_pipeline
from risk_model import train_model, predict_risk
from anonymization import anonymization_pipeline
from visualization import (
    plot_risk_distribution,
    compare_risk,
    plot_feature_importance
)


# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "D:/GitHub/Project/ML-based Privacy Risk Model/DataSet/day_1.parquet"


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def run_pipeline():

    print("🔹 Loading dataset...")
    df = pd.read_parquet(DATA_PATH)

    print("🔹 Running feature engineering...")
    df = feature_engineering_pipeline(df)

    print("🔹 Training ML model...")
    model = train_model(df)

    print("🔹 Predicting risk scores...")
    df = predict_risk(model, df)

    # Save copy BEFORE anonymization
    df_before = df.copy()

    print("🔹 Applying anonymization...")
    df_after = anonymization_pipeline(df)

    print("🔹 Generating visualizations...")

    # Risk distributions
    plot_risk_distribution(df_before, "Risk Distribution - Before")
    plot_risk_distribution(df_after, "Risk Distribution - After")

    # Comparison
    compare_risk(df_before, df_after)

    # Feature importance
    feature_names = [
        "group_size",
        "uniqueness_score",
        "rarity_score",
        "location_granularity",
        "time_granularity",
        "sensitive_count"
    ]

    plot_feature_importance(model, feature_names)

    print("🔹 Saving processed dataset...")

    output_path = "C:/Users/dinak/Downloads/app_data/day_1_processed.parquet"
    df_after.to_parquet(output_path, index=False)

    print(f"✅ Process complete. File saved at: {output_path}")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    run_pipeline()
