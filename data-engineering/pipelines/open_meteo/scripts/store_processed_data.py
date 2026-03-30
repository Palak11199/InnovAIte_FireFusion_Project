import shutil
from pathlib import Path

source_file = Path("data/melbourne_weather_cleaned.csv")
output_dir = Path("processed")
output_dir.mkdir(exist_ok=True)

target_file = output_dir / "weather_data_final.csv"

shutil.copy(source_file, target_file)

print(f"Processed data stored at: {target_file}")