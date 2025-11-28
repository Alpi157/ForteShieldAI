"""
retrain_models.py

Перетренировка Fast Gate + Meta Brain challenger-моделей
на основе свежих размеченных кейсов из case_store и offline Feature Store.

Что делает:
  1. Загружает:
      - config/model_registry.json
      - data/processed/features_offline_v11.parquet
      - case_store: data/cases.parquet ИЛИ data/cases.db (SQLite, таблица cases)
  2. Собирает тренировочный датасет:
      - join по docno
      - user_label берётся из case_store (текст: 'fraud' / 'legit' / None)
      - use_for_retrain: если 0 → кейс исключается из выборки
      - итоговая метка:
          * если user_label в {fraud, legit} и use_for_retrain != 0 →
                label = 1 для fraud, 0 для legit
          * иначе → label = target из offline Feature Store
  3. Обучает challenger-версии:
      - Fast Gate challenger: CatBoost с тем же списком фич,
        что и у champion (features_path из model_registry)
      - Meta Brain challenger: LogisticRegression pipeline,
        с тем же meta-списком (features_path meta_brain.champion)
  4. Считает OOF-метрики (ROC-AUC, PR-AUC, F1 по лучшему порогу),
     сравнивает с champion-метриками из model_registry.
  5. Сохраняет:
      - новые веса моделей в models/
      - отчёт в reports/retrain_report_<timestamp>.json
      - обновлённый config/model_registry.json с заполненным challenger.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Пути проекта
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
PROC_PATH = DATA_ROOT / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "config"
REPORTS_PATH = PROJECT_ROOT / "reports"

FEATURES_PARQUET = PROC_PATH / "features_offline_v11.parquet"
REGISTRY_PATH = CONFIG_PATH / "model_registry.json"

MODELS_PATH.mkdir(parents=True, exist_ok=True)
CONFIG_PATH.mkdir(parents=True, exist_ok=True)
REPORTS_PATH.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Вспомогательные функции
# -----------------------------
def now_iso() -> str:
    """Текущее время в ISO 8601 без микросекунд."""
    return datetime.utcnow().replace(microsecond=0).isoformat()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_case_store() -> pd.DataFrame:
    """
    Загружаем case_store.

    Приоритет:
      1) data/cases.parquet
      2) data/cases.db (SQLite, таблица cases)
    """
    parquet_path = DATA_ROOT / "cases.parquet"
    sqlite_path = DATA_ROOT / "cases.db"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        print(f"[OK] case_store: прочитан {parquet_path}, строк={len(df):,}")
        return df

    if sqlite_path.exists():
        conn = sqlite3.connect(sqlite_path)
        try:
            df = pd.read_sql("SELECT * FROM cases", conn)
        finally:
            conn.close()
        print(f"[OK] case_store: прочитан {sqlite_path} (таблица cases), строк={len(df):,}")
        return df

    raise FileNotFoundError(
        "Не найден case_store: ожидается data/cases.parquet или data/cases.db (таблица cases)."
    )


def ensure_column(df: pd.DataFrame, col_name: str, default=None):
    """Гарантирует наличие колонки col_name в df."""
    if col_name not in df.columns:
        df[col_name] = default
    return df


def eval_at_threshold(y_true, y_proba, threshold: float):
    """Метрики при заданном пороге для бинарной классификации."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def pick_best_f1_metrics(y_true, y_proba, num_thresholds: int = 501):
    """Перебираем сетку порогов и выбираем строку с максимальным F1."""
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    rows = [eval_at_threshold(y_true, y_proba, t) for t in thresholds if t > 0.0]
    df = pd.DataFrame(rows)
    best = df.loc[df["f1"].idxmax()]
    return best.to_dict()


def map_user_label_to_int(v):
    """
    Маппинг текстового user_label из case_store в {0,1}:

      'fraud' / '1' -> 1
      'legit' / '0' -> 0
      иначе -> np.nan
    """
    if v is None:
        return np.nan
    # если уже число / bool
    if isinstance(v, (int, bool, np.integer)):
        iv = int(v)
        if iv == 0:
            return 0
        if iv == 1:
            return 1
        return np.nan

    s = str(v).strip().lower()
    if s in ("1", "fraud", "fraudulent", "fraud_case"):
        return 1
    if s in ("0", "legit", "genuine", "normal", "clean"):
        return 0
    return np.nan


def map_use_for_retrain_flag(v):
    """
    Приводим use_for_retrain к 0/1:

      None / NaN -> 1 (по умолчанию используем)
      bool       -> 1/0
      int/str    -> 0, если значение == 0; иначе 1
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 1
    if isinstance(v, bool):
        return int(v)
    try:
        iv = int(v)
        return 0 if iv == 0 else 1
    except Exception:
        return 1


def enrich_champion_metrics(
    orig_metrics: dict | None,
    dataset_name: str,
    n_samples: int,
    n_pos: int,
    n_neg: int,
) -> dict:
    """
    Дополняет champion.metrics недостающими полями:
      - dataset
      - n_samples
      - n_pos
      - n_neg
      - baseline_pr_auc (pos_frac)
    Если какие-то из этих ключей уже есть — не трогаем их.
    """
    if orig_metrics is None:
        orig_metrics = {}
    m = dict(orig_metrics)

    if "dataset" not in m:
        m["dataset"] = dataset_name
    if "n_samples" not in m:
        m["n_samples"] = int(n_samples)
    if "n_pos" not in m:
        m["n_pos"] = int(n_pos)
    if "n_neg" not in m:
        m["n_neg"] = int(n_neg)
    if "baseline_pr_auc" not in m:
        pos_frac = n_pos / max(1, n_samples)
        m["baseline_pr_auc"] = float(pos_frac)

    return m


# -----------------------------
# 1. Загрузка registry и feature store
# -----------------------------
if not REGISTRY_PATH.exists():
    raise FileNotFoundError(f"Не найден model_registry.json: {REGISTRY_PATH}")

registry = load_json(REGISTRY_PATH)
print(f"[OK] Прочитан model_registry.json из {REGISTRY_PATH}")

if not FEATURES_PARQUET.exists():
    raise FileNotFoundError(
        f"Не найден offline Feature Store: {FEATURES_PARQUET}. "
        "Сначала нужно собрать features_offline_v11.parquet."
    )

print(f"Читаем offline Feature Store: {FEATURES_PARQUET}")
features_df = pd.read_parquet(FEATURES_PARQUET)
print(f"  строк: {len(features_df):,}, колонок: {features_df.shape[1]}")

if "target" not in features_df.columns or "docno" not in features_df.columns:
    raise ValueError("В Feature Store должны быть хотя бы колонки 'target' и 'docno'.")

features_df["target"] = features_df["target"].astype(int)


# -----------------------------
# 2. Загрузка case_store и сбор тренировочной выборки
# -----------------------------
case_df = load_case_store()

case_df = ensure_column(case_df, "user_label", default=None)
case_df = ensure_column(case_df, "use_for_retrain", default=1)
case_df = ensure_column(case_df, "created_at", default=None)

if "docno" not in case_df.columns:
    raise ValueError("В case_store должна быть колонка 'docno' для join'а с Feature Store.")

# Маппинг user_label (TEXT) → user_label_num (0/1/NaN)
case_df["user_label_num"] = case_df["user_label"].apply(map_user_label_to_int)

# Маппинг use_for_retrain → флаг 0/1
case_df["use_for_retrain_flag"] = case_df["use_for_retrain"].apply(map_use_for_retrain_flag)

# Оставляем только нужные для join колонки
case_join_cols = ["docno", "user_label_num", "use_for_retrain_flag", "created_at"]
case_df = case_df[case_join_cols]

print("\nJoin Feature Store + case_store по docno...")
df = features_df.merge(case_df, on="docno", how="left", suffixes=("", "_case"))

# Фильтр по use_for_retrain_flag:
#   строки с use_for_retrain_flag == 0 — исключаем из тренировки;
#   NaN (нет кейса в case_store) — оставляем.
mask_keep = df["use_for_retrain_flag"].isna() | (df["use_for_retrain_flag"] != 0)
df = df[mask_keep].copy()

# Итоговая метка:
#   если analyst поставил user_label_num (0/1) — берём его,
#   иначе используем исходный target.
df["label"] = np.where(df["user_label_num"].isin([0, 1]), df["user_label_num"], df["target"])
df = df[df["label"].isin([0, 1])].copy()
df["label"] = df["label"].astype(int)

n_total = len(df)
n_pos = int((df["label"] == 1).sum())
n_neg = int((df["label"] == 0).sum())
n_from_labels = int(df["user_label_num"].isin([0, 1]).sum())
n_from_target = int(n_total - n_from_labels)

print(
    f"После join + use_for_retrain-фильтра: строк={n_total:,}, "
    f"pos={n_pos}, neg={n_neg}, pos_frac={n_pos / max(1, n_total):.5f}"
)
print(
    f"  - от аналитиков (user_label): {n_from_labels} строк "
    f"({n_from_labels / max(1, n_total):.3%})"
)
print(
    f"  - от исторического target: {n_from_target} строк "
    f"({n_from_target / max(1, n_total):.3%})"
)

# Признак "новизны" кейса (для отчёта)
def parse_dt(s):
    if s is None:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


fast_champion = registry.get("fast_gate", {}).get("champion", {})
trained_until_str = fast_champion.get("trained_until")
trained_until_dt = parse_dt(trained_until_str)

# Считаем created_at как datetime (если есть), иначе используем transdatetime из Feature Store
df["_created_at_dt"] = pd.to_datetime(df.get("created_at"), errors="coerce")
if "transdatetime" in df.columns:
    df["_trans_dt"] = pd.to_datetime(df["transdatetime"], errors="coerce")
else:
    df["_trans_dt"] = pd.NaT

df["_event_time"] = df[["_created_at_dt", "_trans_dt"]].max(axis=1)

if trained_until_dt is not None:
    df["_is_new_since_champion"] = df["_event_time"] > trained_until_dt
    n_new = int(df["_is_new_since_champion"].sum())
else:
    df["_is_new_since_champion"] = False
    n_new = 0

print(f"Из них новых кейсов (event_time > trained_until champion Fast Gate): {n_new}")

# dataset-метаданные (используются и для challenger, и для champion)
DATASET_NAME = "feature_store_join_cases"


# -----------------------------
# 3. Подготовка списков фич Fast Gate и Meta Brain
# -----------------------------
def load_feature_list(path: Path) -> list:
    """Гибкая загрузка списка фич из JSON."""
    data = load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["features", "feature_names", "meta_features", "columns", "feature_list"]:
            if key in data and isinstance(data[key], list):
                return data[key]
        for v in data.values():
            if isinstance(v, list):
                return v
    raise ValueError(f"Не удалось интерпретировать json как список фич: {path}")


# --- Fast Gate ---
fast_info = registry.get("fast_gate", {}).get("champion")
if fast_info is None:
    raise ValueError("В registry нет блока fast_gate.champion — нечего переобучать.")

fast_features_path = CONFIG_PATH / fast_info["features_path"]
if not fast_features_path.exists():
    fast_features_path = PROJECT_ROOT / fast_info["features_path"]

fast_feature_list = load_feature_list(fast_features_path)
missing_fast = [c for c in fast_feature_list if c not in df.columns]
if missing_fast:
    raise ValueError(f"[Fast Gate] В df нет колонок фич: {missing_fast[:10]} ...")

print(f"\nFast Gate champion features: {len(fast_feature_list)} фич, schema={fast_features_path}")

# --- Meta Brain ---
meta_info = registry.get("meta_brain", {}).get("champion")
if meta_info is None:
    raise ValueError("В registry нет блока meta_brain.champion — нечего переобучать.")

meta_features_path = CONFIG_PATH / meta_info["features_path"]
if not meta_features_path.exists():
    meta_features_path = PROJECT_ROOT / meta_info["features_path"]

meta_feature_list = load_feature_list(meta_features_path)
missing_meta = [c for c in meta_feature_list if c not in df.columns]
if missing_meta:
    print(
        f" [Meta Brain] В df нет некоторых meta-фич (будут созданы NaN): {missing_meta[:10]} ..."
    )
    for col in missing_meta:
        df[col] = np.nan

print(f"Meta Brain champion features: {len(meta_feature_list)} фич, schema={meta_features_path}")


# -----------------------------
# 4. Обучение Fast Gate challenger (CatBoost, OOF)
# -----------------------------
def train_fast_gate_challenger(df_train: pd.DataFrame, feature_list: list, label_col: str = "label"):
    print("\n================= Fast Gate Challenger: OOF-обучение =================")
    X = df_train[feature_list].copy()
    y = df_train[label_col].astype(int).values

    # Категориальные фичи по типу
    cat_global = [c for c in feature_list if str(df_train[c].dtype) in ("object", "category")]
    for c in cat_global:
        X[c] = X[c].astype(str)

    cat_indices = [feature_list.index(c) for c in cat_global]

    n_samples = len(X)
    n_pos = int((y == 1).sum())
    n_neg = n_samples - n_pos
    print(f"Train set: n={n_samples}, pos={n_pos}, neg={n_neg}, pos_frac={n_pos / max(1, n_samples):.6f}")
    print("Категориальные фичи:", cat_global)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(n_samples, dtype=float)
    fold_metrics = []

    for fold_id, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y[train_idx], y[valid_idx]

        pos_fold = int((y_tr == 1).sum())
        neg_fold = int((y_tr == 0).sum())
        scale_pos_weight = neg_fold / max(1, pos_fold)

        print(
            f"\n[Fast Gate] Fold {fold_id}/5: "
            f"train_pos={pos_fold}, train_neg={neg_fold}, scale_pos_weight={scale_pos_weight:.2f}"
        )

        train_pool = Pool(X_tr, y_tr, cat_features=cat_indices if cat_indices else None)
        valid_pool = Pool(X_val, y_val, cat_features=cat_indices if cat_indices else None)

        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "iterations": 2000,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_seed": 42 + fold_id,
            "border_count": 254,
            "scale_pos_weight": scale_pos_weight,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.5,
            "use_best_model": True,
            "early_stopping_rounds": 100,
            "task_type": "CPU",
            "verbose": 200,
        }

        model_fold = CatBoostClassifier(**params)
        model_fold.fit(train_pool, eval_set=valid_pool)

        proba_val = model_fold.predict_proba(valid_pool)[:, 1]
        oof_pred[valid_idx] = proba_val

        fold_auc = roc_auc_score(y_val, proba_val)
        fold_pr = average_precision_score(y_val, proba_val)
        fold_metrics.append({"fold": fold_id, "roc_auc": fold_auc, "pr_auc": fold_pr})
        print(f"[Fast Gate] Fold {fold_id}: ROC-AUC={fold_auc:.4f}, PR-AUC={fold_pr:.4f}")

    fold_df = pd.DataFrame(fold_metrics)
    print("\n[Fast Gate] CV по фолдам:")
    print(fold_df)

    roc_oof = roc_auc_score(y, oof_pred)
    pr_oof = average_precision_score(y, oof_pred)
    baseline_pr = float((y == 1).mean())
    print(
        f"\n[Fast Gate] OOF ROC-AUC={roc_oof:.4f}, PR-AUC={pr_oof:.4f}, "
        f"baseline PR-AUC={baseline_pr:.6f}"
    )

    best = pick_best_f1_metrics(y, oof_pred, num_thresholds=501)
    print(
        "[Fast Gate] Лучший F1: "
        f"thr={best['threshold']:.3f}, prec={best['precision']:.3f}, "
        f"recall={best['recall']:.3f}, f1={best['f1']:.3f}"
    )

    # Финальная модель на всём датасете
    final_pool = Pool(X, y, cat_features=cat_indices if cat_indices else None)
    final_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=int(1.1 * np.mean([m["roc_auc"] for m in fold_metrics]) * 1000),
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=999,
        border_count=254,
        scale_pos_weight=n_neg / max(1, n_pos),
        bootstrap_type="Bayesian",
        bagging_temperature=0.5,
        task_type="CPU",
        verbose=False,
    )
    final_model.fit(final_pool)

    metrics = {
        "dataset": DATASET_NAME,
        "n_samples": int(n_samples),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "roc_auc": float(roc_oof),
        "pr_auc": float(pr_oof),
        "baseline_pr_auc": float(baseline_pr),
        "best_threshold": float(best["threshold"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "best_f1": float(best["f1"]),
    }

    return final_model, metrics


fast_model_challenger, fast_metrics_challenger = train_fast_gate_challenger(
    df_train=df,
    feature_list=fast_feature_list,
    label_col="label",
)


# -----------------------------
# 5. Обучение Meta Brain challenger (LogReg pipeline, OOF)
# -----------------------------
def train_meta_challenger(df_train: pd.DataFrame, feature_list: list, label_col: str = "label"):
    print("\n================= Meta Brain Challenger: OOF-обучение =================")
    X = df_train[feature_list].copy()
    y = df_train[label_col].astype(int).values

    n_samples = len(X)
    n_pos = int((y == 1).sum())
    n_neg = n_samples - n_pos
    pos_frac = n_pos / max(1, n_samples)

    print(f"Train set Meta: n={n_samples}, pos={n_pos}, neg={n_neg}, pos_frac={pos_frac:.6f}")
    print(f"Число meta-фич: {len(feature_list)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    oof_pred = np.zeros(n_samples, dtype=float)
    fold_metrics = []

    for fold_id, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y[train_idx], y[valid_idx]

        print(
            f"\n[Meta] Fold {fold_id}/5: "
            f"train_pos={int((y_tr == 1).sum())}, train_neg={int((y_tr == 0).sum())}"
        )

        pipe_fold = pipeline
        pipe_fold.fit(X_tr, y_tr)

        proba_val = pipe_fold.predict_proba(X_val)[:, 1]
        oof_pred[valid_idx] = proba_val

        fold_auc = roc_auc_score(y_val, proba_val)
        fold_pr = average_precision_score(y_val, proba_val)
        fold_metrics.append({"fold": fold_id, "roc_auc": fold_auc, "pr_auc": fold_pr})
        print(f"[Meta] Fold {fold_id}: ROC-AUC={fold_auc:.4f}, PR-AUC={fold_pr:.4f}")

    fold_df = pd.DataFrame(fold_metrics)
    print("\n[Meta] CV по фолдам:")
    print(fold_df)

    roc_oof = roc_auc_score(y, oof_pred)
    pr_oof = average_precision_score(y, oof_pred)
    baseline_pr = float(pos_frac)
    print(
        f"\n[Meta] OOF ROC-AUC={roc_oof:.4f}, PR-AUC={pr_oof:.4f}, baseline PR-AUC={baseline_pr:.6f}"
    )

    best = pick_best_f1_metrics(y, oof_pred, num_thresholds=300)
    print(
        "[Meta] Лучший F1: "
        f"thr={best['threshold']:.3f}, prec={best['precision']:.3f}, "
        f"recall={best['recall']:.3f}, f1={best['f1']:.3f}"
    )

    # Финальная meta-модель на всём датасете
    final_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    final_pipeline.fit(X, y)

    # importance = |коэффициенты логистики|
    clf = final_pipeline.named_steps["clf"]
    coefs = clf.coef_[0]
    importance = np.abs(coefs)
    meta_importance_df = pd.DataFrame(
        {"feature": feature_list, "importance": importance, "coef": coefs}
    ).sort_values("importance", ascending=False)

    metrics = {
        "dataset": DATASET_NAME,
        "n_samples": int(n_samples),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "roc_auc": float(roc_oof),
        "pr_auc": float(pr_oof),
        "baseline_pr_auc": float(baseline_pr),
        "best_threshold": float(best["threshold"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "best_f1": float(best["f1"]),
        "top_features": meta_importance_df.head(10).to_dict(orient="records"),
    }

    return final_pipeline, metrics


meta_model_challenger, meta_metrics_challenger = train_meta_challenger(
    df_train=df,
    feature_list=meta_feature_list,
    label_col="label",
)


# -----------------------------
# 6. Сохранение challenger-моделей
# -----------------------------
timestamp_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

fast_model_rel_path = f"models/catboost_fast_gate_challenger_{timestamp_tag}.cbm"
fast_model_path = PROJECT_ROOT / fast_model_rel_path
fast_model_challenger.save_model(str(fast_model_path))
print(f"\n[OK] Fast Gate challenger сохранён в {fast_model_path}")

meta_model_rel_path = f"models/risk_meta_challenger_{timestamp_tag}.pkl"
meta_model_path = PROJECT_ROOT / meta_model_rel_path
joblib.dump(meta_model_challenger, meta_model_path)
print(f"[OK] Meta Brain challenger сохранён в {meta_model_path}")


# -----------------------------
# 7. Обновление model_registry.json (challenger-блоки)
# -----------------------------
def get_champion_metrics(reg_model_block: dict):
    m = (reg_model_block or {}).get("metrics") or {}
    return float(m.get("pr_auc", 0.0)), float(m.get("roc_auc", 0.0))


fast_champion_block = registry.get("fast_gate", {}).get("champion", {})
fast_champion_pr, fast_champion_roc = get_champion_metrics(fast_champion_block)

meta_champion_block = registry.get("meta_brain", {}).get("champion", {})
meta_champion_pr, meta_champion_roc = get_champion_metrics(meta_champion_block)

fast_is_better = fast_metrics_challenger["pr_auc"] > fast_champion_pr
meta_is_better = meta_metrics_challenger["pr_auc"] > meta_champion_pr

trained_until_new = df["_event_time"].max()
if pd.isna(trained_until_new):
    trained_until_new_iso = now_iso()
else:
    trained_until_new_iso = trained_until_new.to_pydatetime().replace(microsecond=0).isoformat()

# Обогащаем champion.metrics (Fast Gate и Meta Brain) датасетной статистикой
fast_champion_metrics_orig = fast_champion_block.get("metrics") or {}
meta_champion_metrics_orig = meta_champion_block.get("metrics") or {}

fast_champion_metrics_full = enrich_champion_metrics(
    fast_champion_metrics_orig,
    dataset_name=DATASET_NAME,
    n_samples=n_total,
    n_pos=n_pos,
    n_neg=n_neg,
)
meta_champion_metrics_full = enrich_champion_metrics(
    meta_champion_metrics_orig,
    dataset_name=DATASET_NAME,
    n_samples=n_total,
    n_pos=n_pos,
    n_neg=n_neg,
)

# Обновляем champion.metrics в registry, чтобы там тоже были все ключи
if "fast_gate" in registry and "champion" in registry["fast_gate"]:
    registry["fast_gate"]["champion"]["metrics"] = fast_champion_metrics_full
if "meta_brain" in registry and "champion" in registry["meta_brain"]:
    registry["meta_brain"]["champion"]["metrics"] = meta_champion_metrics_full

# Обновляем challenger для fast_gate
registry.setdefault("fast_gate", {})
registry["fast_gate"]["challenger"] = {
    "version": f"challenger_{timestamp_tag}",
    "model_path": fast_model_rel_path,
    "features_path": fast_info["features_path"],  # та же схема фич, что у champion
    "trained_until": trained_until_new_iso,
    "metrics": fast_metrics_challenger,
}

# Обновляем challenger для meta_brain
registry.setdefault("meta_brain", {})
registry["meta_brain"]["challenger"] = {
    "version": f"challenger_{timestamp_tag}",
    "model_path": meta_model_rel_path,
    "features_path": meta_info["features_path"],  # та же схема meta-фич
    "trained_until": trained_until_new_iso,
    "metrics": meta_metrics_challenger,
}

registry["updated_at"] = now_iso()
save_json(registry, REGISTRY_PATH)
print(f"\n[OK] model_registry.json обновлён (challenger для fast_gate и meta_brain).")


# -----------------------------
# 8. Отчёт по retrain (reports/)
# -----------------------------
report = {
    "generated_at": now_iso(),
    "training_stats": {
        "n_total": int(n_total),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "pos_frac": float(n_pos / max(1, n_total)),
        "n_new_since_fast_champion": int(n_new),
        "n_from_user_labels": int(n_from_labels),
        "n_from_target": int(n_from_target),
    },
    "fast_gate": {
        "champion": {
            "version": fast_champion_block.get("version"),
            "trained_until": fast_champion_block.get("trained_until"),
            "metrics": fast_champion_metrics_full,
        },
        "challenger": {
            "version": f"challenger_{timestamp_tag}",
            "model_path": fast_model_rel_path,
            "metrics": fast_metrics_challenger,
        },
        "is_challenger_better_pr_auc": bool(fast_is_better),
    },
    "meta_brain": {
        "champion": {
            "version": meta_champion_block.get("version"),
            "trained_until": meta_champion_block.get("trained_until"),
            "metrics": meta_champion_metrics_full,
        },
        "challenger": {
            "version": f"challenger_{timestamp_tag}",
            "model_path": meta_model_rel_path,
            "metrics": meta_metrics_challenger,
        },
        "is_challenger_better_pr_auc": bool(meta_is_better),
    },
}

report_path = REPORTS_PATH / f"retrain_report_{timestamp_tag}.json"
save_json(report, report_path)

print(f"\n[OK] Retrain-отчёт сохранён в {report_path}")
print("\nГотово. Теперь backend может:")
print("  - через /reload_models перечитать model_registry.json")
print("  - в UI показать сравнение champion vs challenger и дать кнопки Promote / Rollback.")
print("  - retrain учитывает user_label ('fraud'/'legit') и use_for_retrain из case_store.")
