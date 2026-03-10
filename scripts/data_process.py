import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = BASE_DIR / "data" / "raw" / "GlobalWeatherRepository.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

df = pd.read_csv(RAW_FILE)

# it shows no. of rows and cols
print("Original shape:", df.shape)

# 1. Remove duplicates
df.drop_duplicates(inplace=True)

# 2. Handle missing values
df.ffill(inplace=True)

# 3. Convert date columns
date_cols = [col for col in df.columns if "date" in col.lower() or "updated" in col.lower()]
date_col = date_cols[0]  
# Use first detected date column
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")


# 4. Drop remaining null rows
df.dropna(inplace=True)

print("Cleaned shape:", df.shape)

# Save cleaned dataset
output_file = PROCESSED_DIR / "cleaned_weather_data.csv"
df.to_csv(output_file, index=False)

print("Cleaned data saved to:", output_file)

# Aggregate and Filter Data
# Group and aggregate
# Check if required columns exist
df["year_month"] = df[date_col].dt.to_period("M")

# Required temperature columns
required_cols = ["temperature_celsius", "temperature_fahrenheit"]

monthly_temp_stats = (
    df.groupby("year_month")[required_cols]
      .agg(["mean", "median", "min", "max"])
      .reset_index()
)

monthly_output = PROCESSED_DIR / "monthly_temperature_stats.csv"
monthly_temp_stats.to_csv(monthly_output, index=False)

print("Monthly temperature statistics saved to:", monthly_output)

print("\nPreview of aggregated data:")
print(monthly_temp_stats.head())


# Milestone - 2 Core Analysis
print("Milestone - 2")
# Average temperature by country
country_avg_temp = df.groupby("country")["temperature_celsius"].mean()
print("\n ***Average Temperature by Country ***")
print(country_avg_temp.to_string())

# Monthly average temperature
monthly_avg_temp = df.groupby("year_month")["temperature_celsius"].mean()

# to calculate multiple stats at once
country_stats = df.groupby("country").agg({
    "temperature_celsius": ["mean", "min", "max"],
    "humidity": ["mean"],
    "wind_kph": ["mean", "max"]
})
print("\n*** Country Weather Statistics ***")
print(country_stats.head(10))

# To compare each row against group average
df["country_avg_temp"] = df.groupby("country")["temperature_celsius"].transform("mean")
df["temp_anomaly"] = df["temperature_celsius"] - df["country_avg_temp"]
print("\n*** Temperature Anomaly Sample ***")
print(df[["country", "temperature_celsius", "temp_anomaly"]].head())

# 7-days rolling average
df = df.sort_values(date_col)
df["rolling_7day_temp"] = df["temperature_celsius"].rolling(window=7).mean().reset_index(level=0, drop=True)
print("\n*** 7-Day Rolling Average ***")
print(df[[date_col, "temperature_celsius", "rolling_7day_temp"]].head(15))

# to detect extreme weather thresholds
# 95th percentile temperature
threshold = df["temperature_celsius"].quantile(0.95)
print("\n95th Percentile Temperature:", threshold)

extreme_heat_days = df[df["temperature_celsius"] > threshold]
print("\n*** Extreme Heat Days Sample ***")
print(extreme_heat_days[[date_col, "country", "temperature_celsius"]].head())

# correlation
print("\n*** Correlation Matrix ***")
corr = df[["temperature_celsius", "humidity", "wind_kph"]].corr()
print(corr)

# temporal data handling
df["year"] = df[date_col].dt.year
df["month"] = df[date_col].dt.month