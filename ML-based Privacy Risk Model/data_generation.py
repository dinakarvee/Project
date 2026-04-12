import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
ROWS_PER_DAY = 1_000_000
NUM_DAYS = 7

OUTPUT_PATH = r"D:\GitHub\Project\ML-based Privacy Risk Model\DataSet\\"

np.random.seed(42)

# -----------------------------
# PRODUCT SETUP
# -----------------------------
categories = {
    "game": 80,
    "productivity": 50,
    "social": 40,
    "finance": 30,
    "health": 30,
    "education": 20
}

product_list = []

for category, count in categories.items():
    for i in range(count):
        product_list.append({
            "productid": f"{category[:3].upper()}_{i:03d}",
            "product_name": f"{category}_app_{i:03d}",
            "app_category": category
        })

products_df = pd.DataFrame(product_list)

# skewed product distribution
product_weights = np.random.zipf(a=2, size=len(products_df))
product_weights = product_weights / product_weights.sum()

# -----------------------------
# GEO SETUP
# -----------------------------
countries = ["US", "IN", "DE", "UK", "CA"]
country_probs = [0.35, 0.30, 0.15, 0.10, 0.10]

states_per_country = {
    "US": ["CA", "TX", "NY", "FL", "WA", "IL", "AZ", "GA"],
    "IN": ["MH", "DL", "KA", "TN", "GJ", "UP", "RJ", "WB"],
    "DE": ["BW", "BY", "HE", "HH", "NI", "NW", "RP", "SN"],
    "UK": ["ENG", "SCT", "WLS", "NIR"],
    "CA": ["ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB"]
}

cities = ["CityA", "CityB", "CityC", "CityD", "CityE"]

# -----------------------------
# GENERATORS (VECTOR)
# -----------------------------
def generate_ids(n):
    chars = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"))
    CVRoot = ["".join(np.random.choice(chars, 22)) for _ in range(n)]
    device_id = ["g:" + "".join(np.random.choice(list("0123456789"), 18)) for _ in range(n)]
    return CVRoot, device_id


def generate_geo(n):
    country = np.random.choice(countries, size=n, p=country_probs)

    state = np.array([
        np.random.choice(states_per_country[c]) for c in country
    ])

    city = np.random.choice(cities, size=n)

    return country, state, city


def generate_timestamp(n, base_date):
    hours = np.arange(24)
    weights = np.array([1]*6 + [3]*6 + [5]*6 + [2]*6)
    probs = weights / weights.sum()

    selected_hours = np.random.choice(hours, size=n, p=probs)
    minutes = np.random.randint(0, 60, n)
    seconds = np.random.randint(0, 60, n)

    timestamps = [
        datetime.combine(base_date, datetime.min.time()) +
        timedelta(hours=int(h), minutes=int(m), seconds=int(s))
        for h, m, s in zip(selected_hours, minutes, seconds)
    ]

    return timestamps


def generate_payment(n):
    probs = np.random.rand(n)

    payment = np.zeros(n)

    payment[(probs >= 0.7) & (probs < 0.85)] = np.random.uniform(1, 10, np.sum((probs >= 0.7) & (probs < 0.85)))
    payment[(probs >= 0.85) & (probs < 0.95)] = np.random.uniform(10, 50, np.sum((probs >= 0.85) & (probs < 0.95)))
    payment[(probs >= 0.95)] = np.random.uniform(50, 200, np.sum((probs >= 0.95)))

    return np.round(payment, 2)


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def generate_day_data(day_index):
    print(f"⚡ Generating day {day_index+1}...")

    base_date = datetime(2026, 4, 1) + timedelta(days=day_index)

    n = ROWS_PER_DAY

    # product selection
    product_idx = np.random.choice(len(products_df), size=n, p=product_weights)
    selected_products = products_df.iloc[product_idx].reset_index(drop=True)

    # vector generation
    CVRoot, device_id = generate_ids(n)
    country, state, city = generate_geo(n)
    timestamps = generate_timestamp(n, base_date)
    payment = generate_payment(n)

    df = pd.DataFrame({
        "date": [base_date.date()] * n,
        "timestamp": timestamps,
        "CVRoot": CVRoot,
        "device_id": device_id,
        "device_type": np.random.choice(["mobile", "desktop", "tablet"], n, p=[0.7, 0.25, 0.05]),
        "productid": selected_products["productid"],
        "product_name": selected_products["product_name"],
        "app_category": selected_products["app_category"],
        "country": country,
        "state": state,
        "city": city,
        "IPregion": country,
        "Market": country,
        "language": np.random.choice(["EN", "OTHER"], n, p=[0.6, 0.4]),
        "user_age_group": np.random.choice(["18-25", "26-35", "36-50", "50+"], n, p=[0.3, 0.4, 0.2, 0.1]),
        "install_hresult": [hex(x) for x in np.random.randint(10**6, 10**8, n)],
        "purchase_hresult": [hex(x) for x in np.random.randint(10**6, 10**8, n)],
        "install_success": np.random.choice([True, False], n, p=[0.85, 0.15]),
        "payment_amount": payment,
        "payment_method": np.random.choice(["card", "wallet", "upi", "none"], n, p=[0.4, 0.2, 0.2, 0.2]),
        "session_duration": np.random.randint(10, 5000, n),
        "AppVersion": [f"{x}.1401.5.0" for x in np.random.randint(20000, 30000, n)],
        "OSversion": [f"10.0.{x}" for x in np.random.randint(20000, 30000, n)]
    })

    return df


def save_data():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    for day in range(NUM_DAYS):
        df = generate_day_data(day)
        file_path = os.path.join(OUTPUT_PATH, f"day_{day+1}.parquet")
        df.to_parquet(file_path, index=False)
        print(f"✅ Saved: {file_path}")


if __name__ == "__main__":
    save_data()
