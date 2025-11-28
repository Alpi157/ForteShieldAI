import requests
from datetime import datetime, timedelta
import random

BACKEND_URL = "http://127.0.0.1:8000"

def generate_micro_fraud_pattern():
    base_time = datetime.now()
    cst_id = "redteam_user"
    direction = "redteam_dir"

    # 5 мелких транзакций
    txs = []
    for i in range(5):
        txs.append({
            "docno": 10_000 + i,
            "cst_dim_id": cst_id,
            "direction": direction,
            "amount": random.uniform(1000, 3000),
            "transdatetime": (base_time + timedelta(minutes=i)).isoformat()
        })
    # один крупный cash-out
    txs.append({
        "docno": 10_000 + 5,
        "cst_dim_id": cst_id,
        "direction": direction,
        "amount": 300_000,
        "transdatetime": (base_time + timedelta(minutes=6)).isoformat()
    })
    return txs

def run_scenario(txs, name: str):
    print(f"Running scenario: {name}")
    results = []
    for tx in txs:
        resp = requests.post(f"{BACKEND_URL}/score_transaction", json=tx)
        data = resp.json()
        print(tx["docno"], tx["amount"], "→", data["decision"], data["risk_score_final"])
        results.append((tx, data))
    return results

if __name__ == "__main__":
    txs = generate_micro_fraud_pattern()
    run_scenario(txs, "Micro-fraud + cash-out")
