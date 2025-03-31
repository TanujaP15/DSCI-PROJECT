import pandas as pd

# Load the processed dataset
df = pd.read_csv("processed_dataset.csv")
df.columns = df.columns.str.strip()


# Check first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Check data distribution
print(df.describe())

print(df.isnull().sum().sum())  # Should be 0
print(df["Disease"].value_counts())
