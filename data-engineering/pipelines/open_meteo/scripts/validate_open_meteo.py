import pandas as pd
from pathlib import Path

input_file = Path("data/melbourne_weather_cleaned.csv")

df = pd.read_csv(input_file)

print("=== VALIDATION REPORT ===")

print("\n1. Missing values:")
print(df.isnull().sum())

print("\n2. Duplicate rows:")
print(df.duplicated().sum())

print("\n3. Data types:")
print(df.dtypes)

print("\n4. Range checks:")

if "relative_humidity_2m" in df.columns:
    invalid_humidity = df[
        (df["relative_humidity_2m"] < 0) | (df["relative_humidity_2m"] > 100)
    ]
    print(f"Invalid humidity rows: {len(invalid_humidity)}")

if "wind_speed_10m" in df.columns:
    invalid_wind = df[df["wind_speed_10m"] < 0]
    print(f"Invalid wind speed rows: {len(invalid_wind)}")

if "temperature_2m" in df.columns:
    invalid_temp = df[
        (df["temperature_2m"] < -50) | (df["temperature_2m"] > 60)
    ]
    print(f"Unrealistic temperature rows: {len(invalid_temp)}")

print("\nValidation complete.")