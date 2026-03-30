import pandas as pd
from pathlib import Path

input_file = Path("data/melbourne_weather_raw.csv")
output_file = Path("data/melbourne_weather_cleaned.csv")

df = pd.read_csv(input_file)

# Remove missing values
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

# Convert time column to datetime
df["time"] = pd.to_datetime(df["time"])

df.to_csv(output_file, index=False)

print(f"Saved cleaned data to {output_file}")
print(df.head())