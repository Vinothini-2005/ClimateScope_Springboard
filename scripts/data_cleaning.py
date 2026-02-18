import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(exist_ok=True)

raw_file = list(RAW_DIR.glob("*.csv"))[0]
df = pd.read_csv(raw_file)

print("Original shape:", df.shape)

# 1. Remove duplicates
df.drop_duplicates(inplace=True)

# 2. Handle missing values
df.fillna(method="ffill", inplace=True)

# 3. Convert date columns
for col in df.columns:
    if "date" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

# 4. Drop remaining null rows
df.dropna(inplace=True)

print("Cleaned shape:", df.shape)

# Save cleaned dataset
output_file = PROCESSED_DIR / "cleaned_weather_data.csv"
df.to_csv(output_file, index=False)

print("Cleaned data saved to:", output_file)
