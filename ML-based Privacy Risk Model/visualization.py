import matplotlib.pyplot as plt


# -----------------------------
# RISK DISTRIBUTION
# -----------------------------
def plot_risk_distribution(df, title="Risk Distribution"):
    plt.figure()
    plt.hist(df["risk_score"], bins=50)
    plt.title(title)
    plt.xlabel("Risk Score")
    plt.ylabel("Frequency")
    plt.show()


# -----------------------------
# BEFORE VS AFTER COMPARISON
# -----------------------------
def compare_risk(before_df, after_df):

    before_mean = before_df["risk_score"].mean()
    after_mean = after_df["risk_score"].mean()

    labels = ["Before", "After"]
    values = [before_mean, after_mean]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Average Risk Reduction")
    plt.ylabel("Risk Score")
    plt.show()

    print("Before Risk:", before_mean)
    print("After Risk:", after_mean)
    print("Risk Reduction %:", 
      (df_before["risk_score"].mean() - df_after["risk_score"].mean()) 
      / df_before["risk_score"].mean() * 100)


# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
def plot_feature_importance(model, feature_names):

    importance = model.coef_[0]

    plt.figure()
    plt.barh(feature_names, importance)
    plt.title("Feature Importance (Logistic Regression)")
    plt.xlabel("Coefficient Value")
    plt.show()
