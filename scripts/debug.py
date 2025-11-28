from pathlib import Path
import sys
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features_offline.parquet"

def main():
    docno = int(sys.argv[1]) if len(sys.argv) > 1 else 10355

    print("Загружаю:", FEATURES_PATH)
    df = pd.read_parquet(FEATURES_PATH)
    print("rows:", len(df), "cols:", len(df.columns))

    print("Тип docno:", df["docno"].dtype)
    print("Минимальный/максимальный docno:", df["docno"].min(), df["docno"].max())

    exists = docno in df["docno"].values
    print(f"docno={docno} в parquet:", exists)

    if exists:
        row = df[df["docno"] == docno].T
        print("Строка по docno:")
        print(row)

if __name__ == "__main__":
    main()
