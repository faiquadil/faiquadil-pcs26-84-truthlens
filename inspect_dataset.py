
from pathlib import Path
import pandas as pd

files = {
    "FAKE NEWS": Path("dataset/Fake.csv"),
    "REAL NEWS": Path("dataset/True.csv"),
}

for label, path in files.items():
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)

    if not path.exists():
        print(f"File not found: {path.resolve()}")
        continue

    try:
        df = pd.read_csv(path)

        print("File:", path)
        print("Rows:", df.shape[0])
        print("Columns:", df.shape[1])
        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nMissing values:")
        print(df.isnull().sum().to_string())

        print("\nFirst 2 rows:")
        print(df.head(2).to_string(index=False))

    except Exception as error:
        print("Error:", error)