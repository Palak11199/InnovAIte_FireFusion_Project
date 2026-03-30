import requests
import pandas as pd
from pathlib import Path

# Example location: Melbourne, Victoria
LATITUDE = -37.8136
LONGITUDE = 144.9631

url = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    "&forecast_days=1"
)

response = requests.get(url, timeout=30)
response.raise_for_status()

data = response.json()
hourly = data.get("hourly", {})

df = pd.DataFrame({
    "time": hourly.get("time", []),
    "temperature_2m": hourly.get("temperature_2m", []),
    "relative_humidity_2m": hourly.get("relative_humidity_2m", []),
    "wind_speed_10m": hourly.get("wind_speed_10m", []),
})

df["latitude"] = LATITUDE
df["longitude"] = LONGITUDE

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "melbourne_weather_raw.csv"
df.to_csv(output_file, index=False)

print(f"Saved raw data to {output_file}")
print(df.head())