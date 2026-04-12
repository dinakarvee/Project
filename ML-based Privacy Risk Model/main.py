from ingestion import load_data
from config import DATA_PATH, OUTPUT_PATH

from feature_engineering import feature_engineering_pipeline
from risk_model import train_model, predict_risk
from anonymization import anonymization_pipeline
from visualization import (
    plot_risk_distribution,
    compare_risk,
    plot_feature_importance
)


def run_pipeline():

    # Step 1: Load Data
    df = load_data(DATA_PATH)

    # Step 2: Feature Engineering
    print("🔹 Running feature engineering...")
    df = feature_engineering_pipeline(df)

    # Step 3: Train Model
    print("🔹 Training ML model...")
    model = train_model(df)

    # Step 4: Predict Risk
    print("🔹 Predicting risk scores...")
    df = predict_risk(model, df)

    df_before = df.copy()

    # Step 5: Anonymization
    df_after = anonymization_pipeline(df)

    # Step 6: Visualization
    print("🔹 Generating visualizations...")

    plot_risk_distribution(df_before, "Risk Distribution - Before")
    plot_risk_distribution(df_after, "Risk Distribution - After")

    compare_risk(df_before, df_after)

    feature_names = [
        "group_size",
        "uniqueness_score",
        "rarity_score",
        "location_granularity",
        "time_granularity",
        "sensitive_count"
    ]

    plot_feature_importance(model, feature_names)

    # Step 7: Save Output
    print("🔹 Saving processed dataset...")
    df_after.to_parquet(OUTPUT_PATH, index=False)

    print(f"✅ Process complete. File saved at: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_pipeline()
