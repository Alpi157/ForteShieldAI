# scripts/stream_simulator.py

import pandas as pd
import requests
from time import sleep
from pathlib import Path

BACKEND_URL = "http://127.0.0.1:8000"


def load_transactions(csv_path: Path) -> pd.DataFrame:
    """
    Читаем новый очищенный transactions.csv от участника A.

    Ожидаемый формат:
      sep=','
      колонки: cst_dim_id, transdate, transdatetime, amount, docno, direction, target
    """

    print("Пробую прочитать как CSV через запятую (новый файл от A)...")

    # Сначала пробуем cp1251, если что — падаем на utf-8
    try:
        df = pd.read_csv(csv_path, sep=",", encoding="cp1251")
    except UnicodeDecodeError:
        print("  ⚠️ cp1251 не зашёл, пробую utf-8...")
        df = pd.read_csv(csv_path, sep=",", encoding="utf-8")

    print("  Колонки:", list(df.columns))

    if "transdatetime" not in df.columns:
        raise ValueError(
            "В CSV не найдена колонка 'transdatetime'. "
            f"Текущие колонки: {list(df.columns)}"
        )

    # ---------- Приведение типов ----------

    # Время
    df["transdatetime"] = pd.to_datetime(
        df["transdatetime"].astype(str).str.strip("'\" "),
        errors="coerce",
    )

    # Дата (если нужна где-то дальше — пусть будет аккуратной)
    if "transdate" in df.columns:
        df["transdate"] = pd.to_datetime(
            df["transdate"].astype(str).str.strip("'\" "),
            errors="coerce",
        ).dt.date

    # docno -> числовой
    df["docno"] = pd.to_numeric(df["docno"], errors="coerce")

    # выкидываем строки без времени или docno
    before = len(df)
    df = df.dropna(subset=["transdatetime", "docno"]).reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"  ⚠️ дропнули {before - after} строк без transdatetime/docno")

    df["docno"] = df["docno"].astype(int)

    # cst_dim_id / direction — в строки
    if "cst_dim_id" in df.columns:
        df["cst_dim_id"] = df["cst_dim_id"].astype(str).str.strip()
    if "direction" in df.columns:
        df["direction"] = df["direction"].astype(str).str.strip()

    # сортировка по времени
    df = df.sort_values("transdatetime").reset_index(drop=True)

    print("  ✅ Успешно прочитали CSV, строк после очистки:", len(df))
    return df


def main(mode: str = "fast"):
    # Определяем корень проекта как "папка на уровень выше /scripts"
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "raw" / "transactions.csv"

    print("Ищу файл по пути:", csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV не найден по пути: {csv_path}")

    df = load_transactions(csv_path)

    for _, row in df.iterrows():
        payload = {
            "docno": int(row["docno"]),
            "cst_dim_id": str(row.get("cst_dim_id", "")),
            "direction": str(row.get("direction", "")),
            "amount": float(row["amount"]),
            "transdatetime": row["transdatetime"].isoformat(),
        }

        resp = requests.post(f"{BACKEND_URL}/score_transaction", json=payload)
        print("status=", resp.status_code, resp.text)

        if mode == "slow":
            sleep(0.1)


if __name__ == "__main__":
    # fast = без задержек, slow = с задержкой 0.1с
    main(mode="fast")
