
from pathlib import Path
import pandas as pd

# File paths
fake_path = Path("dataset/Fake.csv")
true_path = Path("dataset/True.csv")
output_path = Path("dataset/combined_fake_news.csv")

# Verify original files
if not fake_path.exists() or not true_path.exists():
    raise FileNotFoundError(
        "Fake.csv or True.csv is missing from the dataset folder."
    )

# Load datasets
fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)

# Check required columns
required_columns = {"title", "text", "subject", "date"}

for name, df in [("Fake.csv", fake_df), ("True.csv", true_df)]:
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

# Add labels: 0 = Fake, 1 = Real
fake_df["label"] = 0
true_df["label"] = 1

# Combine both datasets
combined_df = pd.concat(
    [fake_df, true_df],
    ignore_index=True
)

# Shuffle rows reproducibly
combined_df = combined_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save the combined dataset
combined_df.to_csv(output_path, index=False)

# Verify and report
print("\nDataset combination successful!")
print("Saved to:", output_path.resolve())
print("Total rows:", len(combined_df))
print("Columns:", combined_df.columns.tolist())

print("\nClass distribution:")
print(
    combined_df["label"]
    .map({0: "Fake", 1: "Real"})
    .value_counts()
)

print("\nMissing values:")
print(combined_df.isnull().sum())

print("\nFirst 5 rows:")
print(combined_df.head().to_string(index=False))