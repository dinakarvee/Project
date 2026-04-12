import pandas as pd

def load_data(path):
    print("🔹 Loading dataset...")
    return pd.read_parquet(path)