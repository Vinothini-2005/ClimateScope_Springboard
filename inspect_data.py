import pandas as pd
from pathlib import Path

raw_file = list(Path("data/raw").glob("*.csv"))[0]
df = pd.read_csv(raw_file)

print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nInfo:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nSample rows:")
print(df.head())
