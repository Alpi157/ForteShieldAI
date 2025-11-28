# backend/main.py

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import os
import json
import math
import sys
import subprocess
import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostClassifier, Pool
import joblib

from .schemas import TransactionInput, ScoreResponse
from .policy_engine import PolicyEngine
from .feature_store import OnlineFeatureStore
from .case_store import CaseStore
from . import llm_agent
from .model_loader import ModelRegistryBackend


# ======================
# ЛОГИРОВАНИЕ
# ======================
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("FORTE_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

logger = logging.getLogger(__name__)


# ======================
# ПУТИ / АРТЕФАКТЫ v11
# ======================
BASE_DIR = Path(__file__).resolve().parent.parent

# оффлайн фичестор (индекс docno) — для стабильного онлайна демо
FEATURES_OFFLINE_V11 = BASE_DIR / "data" / "processed" / "features_offline_v11.parquet"

# модели (граф/сесс/ae/seq — по-прежнему статические champion’ы)
MODEL_FAST_V11 = BASE_DIR / "models" / "catboost_fast_gate_v11.cbm"   # fallback
MODEL_GRAPH_V11 = BASE_DIR / "models" / "graph_brain_v11.cbm"
MODEL_SESS_V11 = BASE_DIR / "models" / "session_brain_v11.cbm"
MODEL_AE_META = BASE_DIR / "models" / "tx_autoencoder_meta_v11.pkl"
MODEL_SEQ_META = BASE_DIR / "models" / "seq_meta_v11.pkl"
MODEL_META_VPROD = BASE_DIR / "models" / "risk_meta_vprod.pkl"        # fallback

# схемы фич
CFG_FAST_FEATS = BASE_DIR / "config" / "features_schema_v11.json"
CFG_GRAPH_FEATS = BASE_DIR / "config" / "graph_brain_features_v11.json"
CFG_SESS_FEATS = BASE_DIR / "config" / "session_features_v11.json"
CFG_AE_META_FEATS = BASE_DIR / "config" / "autoencoder_meta_features_v11.json"
CFG_SEQ_META_FEATS = BASE_DIR / "config" / "sequence_meta_features_v11.json"
CFG_META_FEATS = BASE_DIR / "config" / "meta_features_vprod.json"

# пороги
CFG_META_THRESH = BASE_DIR / "config" / "meta_thresholds_vprod.json"
CFG_GRAPH_THRESH = BASE_DIR / "config" / "graph_brain_thresholds_v11.json"
CFG_SESS_THRESH = BASE_DIR / "config" / "session_thresholds_v11.json"
CFG_AE_THRESH = BASE_DIR / "config" / "autoencoder_thresholds_v11.json"

# SHAP (глобальные) — опционально для UI (сейчас не используем напрямую в backend)
SHAP_FAST_SUMMARY = BASE_DIR / "data" / "processed" / "fast_gate_shap_summary_v11.parquet"
SHAP_GRAPH_SUMMARY = BASE_DIR / "data" / "processed" / "graph_brain_shap_summary_v11.parquet"
SHAP_SESS_SUMMARY = BASE_DIR / "data" / "processed" / "session_brain_shap_summary_v11.parquet"

# словарь фич для человекопонятных описаний
FEATURE_DICT_PATH = BASE_DIR / "config" / "feature_dictionary.json"
if FEATURE_DICT_PATH.exists():
    with open(FEATURE_DICT_PATH, "r", encoding="utf-8") as _f:
        FEATURE_DICT: Dict[str, Any] = json.load(_f)
else:
    FEATURE_DICT = {}
    logger.warning("feature_dictionary.json не найден по пути %s", FEATURE_DICT_PATH)


# ================
# APP & SERVICES
# ================
app = FastAPI()

online_store = OnlineFeatureStore()
case_store = CaseStore(str(BASE_DIR / "data" / "cases.db"))
policy_engine = PolicyEngine(config_path=str(CFG_META_THRESH))

NOHIST_BASE_RATE = float(os.getenv("NOHIST_BASE_RATE", "0.02"))

# модельный реестр (Fast Gate + Meta Brain)
model_registry_backend = ModelRegistryBackend()

# shadow-флаги и challenger-модели
FAST_SHADOW_ENABLED: bool = False
META_SHADOW_ENABLED: bool = False

# текущие champion/ challenger модели
cat_fast: Optional[CatBoostClassifier] = None           # Fast Gate champion
meta_vprd = None                                        # Meta Brain champion
cat_fast_challenger: Optional[CatBoostClassifier] = None
meta_vprd_challenger = None


# ==========================
# ЗАГРУЗКА ОФФЛАЙН ФИЧЕСТОРА
# ==========================
if not FEATURES_OFFLINE_V11.exists():
    raise FileNotFoundError(
        f"Не найден оффлайн фичестор: {FEATURES_OFFLINE_V11}. "
        f"Положи features_offline_v11.parquet в data/processed/"
    )

features_df = pd.read_parquet(FEATURES_OFFLINE_V11)
if "docno" not in features_df.columns:
    raise ValueError("В features_offline_v11.parquet отсутствует колонка 'docno'.")
features_df = features_df.set_index("docno")


# ====================
# ЗАГРУЗКА МОДЕЛЕЙ
# ====================
def load_catboost(path: Path) -> CatBoostClassifier:
    if not path.exists():
        raise FileNotFoundError(f"CatBoost модель не найдена: {path}")
    m = CatBoostClassifier()
    m.load_model(str(path))
    return m


def load_joblib(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Joblib-модель не найдена: {path}")
    return joblib.load(path)


# graph / session / ae / seq — пока статические
cat_graph = load_catboost(MODEL_GRAPH_V11)
cat_sess = load_catboost(MODEL_SESS_V11)

ae_meta = load_joblib(MODEL_AE_META)      # Pipeline(Imputer/Scaler + LogisticRegression)
seq_meta = load_joblib(MODEL_SEQ_META)    # Pipeline(Imputer/Scaler + LogisticRegression)


# ====================
# СХЕМЫ ФИЧ (списки)
# ====================
def _read_feature_list(p: Path, keys: Tuple[str, ...]) -> List[str]:
    if not p.exists():
        raise FileNotFoundError(f"Файл схемы фич не найден: {p}")
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        # fallback — первый list в dict
        for v in obj.values():
            if isinstance(v, list):
                return v
    elif isinstance(obj, list):
        return obj
    raise ValueError(f"Не удалось извлечь список фич из {p}")


def _read_cat_indices(p: Path, default: Optional[List[int]] = None) -> List[int]:
    """
    Читает индексы категориальных фич из JSON.
    """
    if default is None:
        default = []
    if not p.exists():
        return default

    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        for key in (
            "cat_feature_indices",
            "categorical_feature_indices",
            "cat_features_idx",
            "cat_idxs",
        ):
            if key in obj:
                v = obj[key]
                if isinstance(v, list):
                    out: List[int] = []
                    for x in v:
                        try:
                            out.append(int(x))
                        except Exception:
                            pass
                    if out:
                        return out
    return default


fast_feature_names = _read_feature_list(
    CFG_FAST_FEATS, ("feature_cols", "features", "feature_names")
)
graph_feature_names = _read_feature_list(
    CFG_GRAPH_FEATS, ("graph_feature_cols", "features")
)
sess_feature_names = _read_feature_list(
    CFG_SESS_FEATS, ("session_feature_cols", "features")
)
ae_meta_feature_names = _read_feature_list(
    CFG_AE_META_FEATS, ("features", "input_features")
)
seq_meta_feature_names = _read_feature_list(
    CFG_SEQ_META_FEATS, ("features", "input_features")
)
meta_feature_names = _read_feature_list(
    CFG_META_FEATS, ("features", "input_features")
)

fast_cat_indices = _read_cat_indices(CFG_FAST_FEATS, default=[55, 56])


# пороги high (опционально)
def _read_threshold(p: Path, key: str = "threshold", default: Optional[float] = None) -> Optional[float]:
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        if key in obj:
            try:
                return float(obj[key])
            except Exception:
                pass
        # поддержка схем типа {"T_graph_high": {"threshold": ...}}
        for k in ("T_graph_high", "high_threshold", "T_ae_high"):
            if k in obj and isinstance(obj[k], dict) and "threshold" in obj[k]:
                try:
                    return float(obj[k]["threshold"])
                except Exception:
                    pass
    return default


T_GRAPH_HIGH = _read_threshold(CFG_GRAPH_THRESH, default=None)
T_SESS_HIGH = _read_threshold(CFG_SESS_THRESH, default=None)
T_AE_HIGH = _read_threshold(CFG_AE_THRESH, default=None)


# ===========================================
# ЧЕЛОВЕКОПОНЯТНЫЕ ОПИСАНИЯ ФИЧ (feature_dict)
# ===========================================
def describe_feature(name: str, value: Any, shap_val: float) -> Dict[str, Any]:
    """
    Обогащает признак метаданными из feature_dictionary.json + SHAP-вкладом.
    """
    meta = FEATURE_DICT.get(name, {})
    direction = "up" if shap_val > 0 else "down"

    high_hint = meta.get("high_risk_hint", "")
    low_hint = meta.get("low_risk_hint", "")

    return {
        "feature": name,
        "raw_value": value,
        "shap": float(shap_val),
        "direction": direction,
        "group": meta.get("group", "other"),
        "label": meta.get("label", name),
        "short_desc": meta.get("short_desc", ""),
        "risk_hint": high_hint if direction == "up" else low_hint,
    }


# =========================
# SHAP на один объект (CB)
# =========================
def shap_top_for_catboost(
    model: CatBoostClassifier,
    feat_names: List[str],
    row_values: List[Any],
    top_k: int = 10,
):
    pool = Pool(data=[row_values], feature_names=feat_names)
    shap_vals = model.get_feature_importance(pool, type="ShapValues")
    phi = shap_vals[0][:-1]
    expected = shap_vals[0][-1]
    pairs = list(zip(feat_names, phi))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    top = [{"feature": n, "contribution": float(v)} for n, v in pairs[:top_k]]
    return top, float(expected)


def catboost_reasons_for_row(
    model: CatBoostClassifier,
    feat_names: List[str],
    row_values: List[Any],
    full_features: Dict[str, Any],
    brain: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Оборачиваем CatBoost TreeSHAP в список человекочитаемых причин.
    """
    try:
        top_raw, _ = shap_top_for_catboost(model, feat_names, row_values, top_k=top_k)
    except Exception:
        return []
    reasons: List[Dict[str, Any]] = []
    for item in top_raw:
        fname = item["feature"]
        shap_val = float(item.get("contribution", 0.0))
        val = full_features.get(fname)
        r = describe_feature(fname, val, shap_val)
        r["brain"] = brain
        reasons.append(r)
    return reasons


# ==========================================
# SHAP-подобные вклады для линейных пайплайнов
# (AE meta / Seq meta / Meta Brain)
# ==========================================
def shap_top_for_linear(
    pipeline,
    feat_names: List[str],
    row: Dict[str, Any],
    brain: str,
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    SHAP-подобные вклады для LogisticRegression внутри Pipeline(imputer, scaler, clf).
    Возвращает (reasons, base_logit, proba).
    """
    if pipeline is None:
        return [], 0.0, 0.0

    try:
        imputer = pipeline.named_steps.get("imputer")
        scaler = pipeline.named_steps.get("scaler")
        clf = pipeline.named_steps.get("clf")
    except Exception:
        return [], 0.0, 0.0

    # 1) X в нужном порядке фич
    X_raw = [[row.get(f, 0.0) for f in feat_names]]
    X_imp = imputer.transform(X_raw) if imputer is not None else X_raw
    X_scaled = scaler.transform(X_imp) if scaler is not None else X_imp

    coefs = getattr(clf, "coef_", None)
    intercept_arr = getattr(clf, "intercept_", None)
    if coefs is None or intercept_arr is None:
        return [], 0.0, 0.0

    coefs = np.array(coefs[0], dtype=float)
    intercept = float(intercept_arr[0])

    x_vec = np.array(X_scaled[0], dtype=float)
    shap_vec = x_vec * coefs  # вклад каждой фичи в логит

    logit = intercept + float(shap_vec.sum())
    proba = 1.0 / (1.0 + math.exp(-logit))

    idxs = list(np.argsort(-np.abs(shap_vec))[:top_k])

    reasons: List[Dict[str, Any]] = []
    for j in idxs:
        fname = feat_names[j]
        val = X_raw[0][j]
        shap_val = float(shap_vec[j])
        r = describe_feature(fname, val, shap_val)
        r["brain"] = brain
        reasons.append(r)

    return reasons, intercept, proba


# ===================
# СКОРОСЧЁТ МОЗГОВ
# ===================
def score_fast(row: Dict[str, Any]) -> Tuple[float, List[Any]]:
    """
    Скоринг Fast Gate champion.
    """
    if cat_fast is None:
        raise RuntimeError("Fast Gate модель ещё не загружена (cat_fast is None)")

    X: List[Any] = []
    for idx, feat in enumerate(fast_feature_names):
        v = row.get(feat)

        if idx in fast_cat_indices:  # категориальные фичи
            if v is None:
                X.append("NA")
            else:
                X.append(str(v))
        else:
            if v is None or v == "":
                X.append(0.0)
            else:
                try:
                    X.append(float(v))
                except Exception:
                    X.append(0.0)

    p = float(cat_fast.predict_proba([X])[0][1])
    return p, X


def score_graph(row: Dict[str, Any]) -> Tuple[float, List[float]]:
    X = [float(row.get(f, 0.0)) for f in graph_feature_names]
    p = float(cat_graph.predict_proba([X])[0][1])
    return p, X


def score_sess(row: Dict[str, Any]) -> Tuple[float, List[Any]]:
    X: List[Any] = []
    for f in sess_feature_names:
        v = row.get(f)
        if isinstance(v, (int, float)) or v is None:
            X.append(v if v is not None else 0.0)
        else:
            X.append(str(v))
    p = float(cat_sess.predict_proba([X])[0][1])
    return p, X


def score_ae_meta(row: Dict[str, Any]) -> float:
    X = [[row.get(f, 0.0) for f in ae_meta_feature_names]]
    try:
        if hasattr(ae_meta, "predict_proba"):
            return float(ae_meta.predict_proba(X)[0][1])
        return float(ae_meta.predict(X)[0])
    except Exception:
        return 0.0


def score_seq_meta(row: Dict[str, Any]) -> Tuple[float, int]:
    X = [[row.get(f, 0.0) for f in seq_meta_feature_names]]
    try:
        if hasattr(seq_meta, "predict_proba"):
            p = float(seq_meta.predict_proba(X)[0][1])
        else:
            p = float(seq_meta.predict(X)[0])
    except Exception:
        p = 0.0
    hist_len = int(row.get("seq_hist_len_v11", row.get("seq_hist_len", 0)) or 0)
    return p, hist_len


def score_meta_vprod(row: Dict[str, Any], parts: Dict[str, float], model=None) -> float:
    """
    Скоринг Meta Brain (champion по умолчанию, можно передать другой model для challenger/shadow).
    """
    m = model or meta_vprd
    if m is None:
        raise RuntimeError("Meta Brain модель ещё не загружена (meta_vprd is None)")

    z: Dict[str, float] = {**{k: row.get(k, 0.0) for k in meta_feature_names}, **parts}
    X = [[z.get(f, 0.0) for f in meta_feature_names]]
    try:
        if hasattr(m, "predict_proba"):
            return float(m.predict_proba(X)[0][1])
        return float(m.predict(X)[0])
    except Exception:
        # fallback — хотя бы риск по Fast Gate
        return parts.get("risk_fast", 0.0)


# ====================
# ВСПОМОГАТЕЛЬНОЕ
# ====================
def bool_flag(value: Optional[float], thr: Optional[float]) -> int:
    if value is None or thr is None:
        return 0
    try:
        return int(float(value) >= float(thr))
    except Exception:
        return 0


# ====================
# FASTAPI EVENTS
# ====================
@app.on_event("startup")
def startup_event() -> None:
    """
    На старте приложения загружаем champion Fast Gate + Meta Brain
    через model_registry.json. Если что-то пошло не так — используем fallback v11/vprod.
    Также подтягиваем challenger-модели и флаги shadow_enabled.
    """
    global cat_fast, meta_vprd, cat_fast_challenger, meta_vprd_challenger
    global FAST_SHADOW_ENABLED, META_SHADOW_ENABLED

    try:
        model_registry_backend.load_champion()
        cat_fast = model_registry_backend.fast_gate
        meta_vprd = model_registry_backend.meta_brain

        cat_fast_challenger = model_registry_backend.fast_gate_challenger
        meta_vprd_challenger = model_registry_backend.meta_brain_challenger

        FAST_SHADOW_ENABLED = model_registry_backend.fast_gate_shadow_enabled
        META_SHADOW_ENABLED = model_registry_backend.meta_brain_shadow_enabled

        logger.info(
            "Loaded models from registry: "
            "FastGate champion=%s, challenger=%s, shadow_enabled=%s; "
            "MetaBrain champion=%s, challenger=%s, shadow_enabled=%s",
            model_registry_backend.current_fast_version,
            model_registry_backend.challenger_fast_version,
            FAST_SHADOW_ENABLED,
            model_registry_backend.current_meta_version,
            model_registry_backend.challenger_meta_version,
            META_SHADOW_ENABLED,
        )
    except Exception as e:
        logger.exception(
            "Failed to load models from registry, fallback to fixed MODEL_FAST_V11 / MODEL_META_VPROD: %s",
            e,
        )
        # fallback-модели
        cat_fast = load_catboost(MODEL_FAST_V11)
        meta_vprd = load_joblib(MODEL_META_VPROD)

        cat_fast_challenger = None
        meta_vprd_challenger = None
        FAST_SHADOW_ENABLED = False
        META_SHADOW_ENABLED = False

        # чтобы /health не падал по None
        model_registry_backend.current_fast_version = "v11_fallback"
        model_registry_backend.current_meta_version = "vprod_fallback"
        model_registry_backend.challenger_fast_version = None
        model_registry_backend.challenger_meta_version = None
        model_registry_backend.fast_gate_shadow_enabled = False
        model_registry_backend.meta_brain_shadow_enabled = False


# ====================
# Pydantic-схемы для ответов
# ====================
class HealthResponse(BaseModel):
    status: str
    fast_gate_version: Optional[str] = None
    meta_brain_version: Optional[str] = None
    fast_shadow_enabled: bool
    meta_shadow_enabled: bool


class CaseLabelUpdate(BaseModel):
    # 'fraud' / 'legit' / 'unknown' / None
    user_label: Optional[str] = None
    use_for_retrain: Optional[bool] = None
    status: Optional[str] = None  # 'open' / 'in_review' / 'closed' / ...


# ====================
# ROUTES: базовые
# ====================
@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "ForteShield AI v11 backend is running"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Лёгкий healthcheck + метаданные для UI.
    Bool-флаги shadow_* типизированы как bool, поэтому FastAPI не ругается.
    """
    return HealthResponse(
        status="ok",
        fast_gate_version=model_registry_backend.current_fast_version or "unknown",
        meta_brain_version=model_registry_backend.current_meta_version or "unknown",
        fast_shadow_enabled=bool(FAST_SHADOW_ENABLED),
        meta_shadow_enabled=bool(META_SHADOW_ENABLED),
    )


# ====================
# SCORE TRANSACTION
# ====================
@app.post("/score_transaction", response_model=ScoreResponse)
def score_transaction(tx: TransactionInput) -> ScoreResponse:
    """
    На вход: docno, cst_dim_id, direction, amount, transdatetime.
    Онлайн расчёт: используем оффлайн-фичестор v11 по docno для всех мозгов + velocity из OnlineFeatureStore.

    Champion-модели управляют решением.
    Challenger-модели (если shadow_enabled) считаются в shadow-режиме и логируются в кейс,
    но не влияют на политику.
    """

    # 0) Попытка найти оффлайн-фичи по docno
    try:
        row = features_df.loc[tx.docno]
    except KeyError:
        fallback = float(NOHIST_BASE_RATE)
        policy_result = policy_engine.apply_policy(
            risk_final=fallback,
            features={"fallback_reason": "docno_not_found_in_features_offline_v11"},
            scores={"risk_meta": fallback},
        )
        case_id = case_store.create_case(
            {
                "docno": tx.docno,
                "cst_dim_id": tx.cst_dim_id,
                "direction": tx.direction,
                "amount": tx.amount,
                "transdatetime": tx.transdatetime.isoformat(),
                "risk_fast": None,
                "risk_ae": None,
                "risk_graph": None,
                "risk_seq": None,
                "risk_sess": None,
                "risk_meta": fallback,
                "anomaly_score_emb": None,
                "ae_high": 0,
                "graph_high": 0,
                "seq_high": 0,
                "sess_high": 0,
                "has_seq_history": 0,
                "strategy_used": policy_engine.current_strategy,
                "decision": policy_result["decision"],
                "risk_type": policy_result["risk_type"],
                "status": "new",
                # shadow / разметка
                "user_label": None,
                "use_for_retrain": 1,
                "risk_fast_shadow": None,
                "risk_meta_shadow": None,
            }
        )
        logger.info(
            "Scored transaction without history (docno=%s), fallback risk=%f, decision=%s",
            tx.docno,
            fallback,
            policy_result["decision"],
        )
        return ScoreResponse(
            case_id=case_id,
            risk_score_final=fallback,
            decision=policy_result["decision"],
            risk_type=policy_result["risk_type"],
            strategy_used=policy_engine.current_strategy,
        )

    offline = row.to_dict()

    # 1) velocity онлайн
    online_feats = online_store.update_and_get_features(
        cst_id=tx.cst_dim_id,
        direction=tx.direction,
        amount=tx.amount,
        ts=tx.transdatetime,
    )
    full_features: Dict[str, Any] = {**offline, **online_feats}

    # 2) мозги (champion)
    risk_fast, X_fast = score_fast(full_features)
    risk_graph, X_graph = score_graph(full_features)
    risk_sess, X_sess = score_sess(full_features)
    risk_ae = score_ae_meta(full_features)
    risk_seq, hist_len = score_seq_meta(full_features)

    # 2b) shadow-скоринг challenger-моделей
    risk_fast_shadow: Optional[float] = None
    risk_meta_shadow: Optional[float] = None

    if FAST_SHADOW_ENABLED and cat_fast_challenger is not None:
        try:
            risk_fast_shadow = float(cat_fast_challenger.predict_proba([X_fast])[0][1])
        except Exception as e:
            logger.warning("Ошибка при shadow-скоринге Fast Gate challenger: %s", e)

    # 3) флаги high
    ae_high = bool_flag(risk_ae, T_AE_HIGH)
    graph_high = bool_flag(risk_graph, T_GRAPH_HIGH)
    sess_high = bool_flag(risk_sess, T_SESS_HIGH)
    seq_high = 0
    has_seq_history = int(hist_len > 0)

    # 4) meta vProd (champion)
    parts_for_meta = {
        "risk_fast_oof_v11": full_features.get("risk_fast_oof_v11", risk_fast),
        "risk_ae_oof_v11": full_features.get("risk_ae_oof_v11", risk_ae),
        "graph_brain_oof_v11": full_features.get("graph_brain_oof_v11", risk_graph),
        "risk_seq_oof_v11": full_features.get("risk_seq_oof_v11", risk_seq),
        "risk_sess_oof_v11": full_features.get("risk_sess_oof_v11", risk_sess),
        "risk_fast": risk_fast,
    }
    risk_meta = score_meta_vprod(full_features, parts_for_meta)

    # 4b) Meta Brain challenger shadow
    if META_SHADOW_ENABLED and meta_vprd_challenger is not None:
        try:
            risk_meta_shadow = score_meta_vprod(
                full_features,
                parts_for_meta,
                model=meta_vprd_challenger,
            )
        except Exception as e:
            logger.warning("Ошибка при shadow-скоринге Meta Brain challenger: %s", e)

    if risk_fast_shadow is not None or risk_meta_shadow is not None:
        logger.debug(
            "Shadow scores (docno=%s): fast_shadow=%s (champ=%s), meta_shadow=%s (champ=%s)",
            tx.docno,
            risk_fast_shadow,
            risk_fast,
            risk_meta_shadow,
            risk_meta,
        )

    # 5) политика (champion-скоры)
    policy_result = policy_engine.apply_policy(
        risk_final=risk_meta,
        features=full_features,
        scores={
            "risk_fast": risk_fast,
            "risk_ae": risk_ae,
            "risk_graph": risk_graph,
            "risk_seq": risk_seq,
            "risk_sess": risk_sess,
            "risk_meta": risk_meta,
        },
    )

    # 6) SHAP (per-row) — для отладки, сейчас в ответ не отдаём
    try:
        _top_fast, _ = shap_top_for_catboost(cat_fast, fast_feature_names, X_fast, top_k=10)
    except Exception:
        _top_fast = []
    try:
        _top_graph, _ = shap_top_for_catboost(cat_graph, graph_feature_names, X_graph, top_k=10)
    except Exception:
        _top_graph = []
    try:
        _top_sess, _ = shap_top_for_catboost(cat_sess, sess_feature_names, X_sess, top_k=10)
    except Exception:
        _top_sess = []

    # 7) сохраняем кейс (включая shadow-скоры и поля для retrain)
    case_data = {
        "docno": tx.docno,
        "cst_dim_id": tx.cst_dim_id,
        "direction": tx.direction,
        "amount": tx.amount,
        "transdatetime": tx.transdatetime.isoformat(),
        "risk_fast": risk_fast,
        "risk_ae": risk_ae,
        "risk_graph": risk_graph,
        "risk_seq": risk_seq,
        "risk_sess": risk_sess,
        "risk_meta": risk_meta,
        "anomaly_score_emb": full_features.get("anomaly_score_emb", None),
        "ae_high": ae_high,
        "graph_high": graph_high,
        "seq_high": seq_high,
        "sess_high": sess_high,
        "has_seq_history": has_seq_history,
        "strategy_used": policy_engine.current_strategy,
        "decision": policy_result["decision"],
        "risk_type": policy_result["risk_type"],
        "status": "new",
        # поля для retrain (заполнят аналитики)
        "user_label": None,
        "use_for_retrain": 1,
        # shadow-скоры challenger моделей
        "risk_fast_shadow": risk_fast_shadow,
        "risk_meta_shadow": risk_meta_shadow,
    }
    case_id = case_store.create_case(case_data)

    logger.info(
        "Scored transaction docno=%s case_id=%s meta_risk=%.4f decision=%s strategy=%s",
        tx.docno,
        case_id,
        risk_meta,
        policy_result["decision"],
        policy_engine.current_strategy,
    )

    return ScoreResponse(
        case_id=case_id,
        risk_score_final=risk_meta,
        decision=policy_result["decision"],
        risk_type=policy_result["risk_type"],
        strategy_used=policy_engine.current_strategy,
    )


# ======================
# LLM / SHAP ДЛЯ КЕЙСА
# ======================
def _build_shap_reasons_for_case(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Строим список причин риска по всем мозгам для данного кейса.
    Используем offline Feature Store (features_offline_v11.parquet)
    + текущие champion-модели.
    """
    docno = case.get("docno")
    if docno is None:
        return []

    try:
        offline_series = features_df.loc[docno]
    except Exception:
        # нет оффлайн-фич по этому docno
        return []

    offline = offline_series.to_dict()
    full_features: Dict[str, Any] = dict(offline)

    reasons_all: List[Dict[str, Any]] = []

    # --- CatBoost-мозги (Fast / Graph / Session) ---
    try:
        _, X_fast = score_fast(full_features)
        fast_reasons = catboost_reasons_for_row(
            cat_fast,
            fast_feature_names,
            X_fast,
            full_features,
            brain="fast_gate",
            top_k=7,
        )
        reasons_all.extend(fast_reasons)
    except Exception:
        pass

    try:
        _, X_graph = score_graph(full_features)
        graph_reasons = catboost_reasons_for_row(
            cat_graph,
            graph_feature_names,
            X_graph,
            full_features,
            brain="graph_brain",
            top_k=5,
        )
        reasons_all.extend(graph_reasons)
    except Exception:
        pass

    try:
        _, X_sess = score_sess(full_features)
        sess_reasons = catboost_reasons_for_row(
            cat_sess,
            sess_feature_names,
            X_sess,
            full_features,
            brain="session_brain",
            top_k=5,
        )
        reasons_all.extend(sess_reasons)
    except Exception:
        pass

    # --- Линейные головы (AE / Seq) ---
    try:
        ae_reasons, _, _ = shap_top_for_linear(
            ae_meta,
            ae_meta_feature_names,
            full_features,
            brain="ae_brain",
            top_k=5,
        )
        reasons_all.extend(ae_reasons)
    except Exception:
        pass

    try:
        seq_reasons, _, _ = shap_top_for_linear(
            seq_meta,
            seq_meta_feature_names,
            full_features,
            brain="sequence_brain",
            top_k=5,
        )
        reasons_all.extend(seq_reasons)
    except Exception:
        pass

    # --- Meta Brain ---
    if meta_vprd is not None:
        parts_for_meta = {
            "risk_fast_oof_v11": offline.get("risk_fast_oof_v11", case.get("risk_fast")),
            "risk_ae_oof_v11": offline.get("risk_ae_oof_v11", case.get("risk_ae")),
            "graph_brain_oof_v11": offline.get("graph_brain_oof_v11", case.get("risk_graph")),
            "risk_seq_oof_v11": offline.get("risk_seq_oof_v11", case.get("risk_seq")),
            "risk_sess_oof_v11": offline.get("risk_sess_oof_v11", case.get("risk_sess")),
            "risk_fast": case.get("risk_fast"),
        }
        z_for_meta = {
            **{k: full_features.get(k, 0.0) for k in meta_feature_names},
            **parts_for_meta,
        }
        try:
            meta_reasons, _, _ = shap_top_for_linear(
                meta_vprd,
                meta_feature_names,
                z_for_meta,
                brain="meta_brain",
                top_k=7,
            )
            reasons_all.extend(meta_reasons)
        except Exception:
            pass

    if reasons_all:
        # сортируем по абсолютному SHAP и ограничиваем размер
        reasons_all.sort(key=lambda r: abs(r.get("shap", 0.0)), reverse=True)
        reasons_all = reasons_all[:30]

    return reasons_all


def _build_expl_payload(case: Dict[str, Any], tops: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Payload для LLM / UI: базовая инфа + скоры + причины.
    """
    return {
        "basic_info": {
            "docno": case.get("docno"),
            "cst_dim_id": case.get("cst_dim_id"),
            "direction": case.get("direction"),
            "amount": case.get("amount"),
            "datetime": case.get("transdatetime"),
        },
        "global_scores": {
            "fast_gate": {"score": case.get("risk_fast")},
            "ae_brain": {"score": case.get("risk_ae")},
            "graph_brain": {"score": case.get("risk_graph")},
            "sequence_brain": {"score": case.get("risk_seq")},
            "session_brain": {"score": case.get("risk_sess")},
            "meta": {"score": case.get("risk_meta")},
        },
        "policy": {
            "strategy": case.get("strategy_used"),
            "decision": case.get("decision"),
            "risk_type": case.get("risk_type"),
            "thresholds": policy_engine.get_strategy().get("config", {}),
        },
        "reasons_flat": tops,
    }


@app.post("/llm/explain_case")
def explain_case(case_id: int):
    case = case_store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    shap_reasons = _build_shap_reasons_for_case(case)
    expl_payload = _build_expl_payload(case, shap_reasons)

    text = llm_agent.explain_case_text(
        case,
        shap_top_features=shap_reasons,
        shap_top_groups=[],
        extra_payload=expl_payload,
    )
    return {"explanation": text}


@app.post("/llm/generate_sar")
def generate_sar(case_id: int):
    case = case_store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    shap_reasons = _build_shap_reasons_for_case(case)
    expl_payload = _build_expl_payload(case, shap_reasons)

    sar_text = llm_agent.generate_sar_text(
        case,
        shap_top_features=shap_reasons,
        shap_top_groups=[],
        extra_payload=expl_payload,
    )
    return {"sar": sar_text}


# ======================
# CASES / STRATEGIES API
# ======================
@app.get("/cases")
def list_cases(limit: int = 30) -> List[dict]:
    return case_store.list_cases(limit=limit)


@app.get("/case/{case_id}")
def get_case(case_id: int) -> dict:
    case = case_store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@app.get("/strategy")
def get_strategy():
    return policy_engine.get_strategy()


@app.get("/strategies")
def list_strategies():
    return policy_engine.list_strategies()


@app.post("/strategy/{name}")
def set_strategy(name: str):
    try:
        policy_engine.set_strategy(name)
        logger.info("Decision strategy changed to '%s'", name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return policy_engine.get_strategy()


# ======================
# АННОТАЦИИ / РАЗМЕТКА КЕЙСОВ
# ======================
@app.post("/case/{case_id}/label")
def update_case_label(case_id: int, payload: CaseLabelUpdate):
    """
    Позволяет аналитику обновить разметку кейса:
      - user_label: 'fraud' / 'legit' / 'unknown' / None
      - use_for_retrain: True/False
      - status: статус обработки кейса (open/in_review/closed/...)
    """
    updated = case_store.update_case_label(
        case_id=case_id,
        user_label=payload.user_label,
        use_for_retrain=payload.use_for_retrain,
        status=payload.status,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return updated


# ======================
# MODEL GOVERNANCE API
# ======================
@app.get("/model_registry")
def get_model_registry():
    """
    Возвращает текущий model_registry.json.
    """
    try:
        model_registry_backend.load_registry()
        return model_registry_backend.registry
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retrain_last_report")
def retrain_last_report():
    """
    Отдаёт последний retrain_report_*.json из reports/.
    """
    reports_dir = BASE_DIR / "reports"
    if not reports_dir.exists():
        raise HTTPException(status_code=404, detail="Каталог reports/ не найден")

    reports = sorted(reports_dir.glob("retrain_report_*.json"))
    if not reports:
        raise HTTPException(status_code=404, detail="Отчёты retrain_report_* ещё не созданы")

    last = reports[-1]
    with open(last, "r", encoding="utf-8") as f:
        obj = json.load(f)
    obj["_report_path"] = str(last)
    return obj


@app.post("/retrain")
def retrain_models():
    """
    Запускает retrain_models.py синхронно.
    После завершения читает последний отчёт и возвращает его + stdout/stderr (обрезанные).
    """
    script_path = BASE_DIR / "retrain_models.py"
    if not script_path.exists():
        raise HTTPException(status_code=500, detail=f"retrain_models.py не найден: {script_path}")

    logger.info("Запуск retrain_models.py: %s", script_path)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        logger.exception("Ошибка при запуске retrain_models.py: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка при запуске retrain_models.py: {e}")

    logger.info("retrain_models.py завершился, returncode=%s", result.returncode)

    # читаем последний отчёт (если есть)
    report = None
    reports_dir = BASE_DIR / "reports"
    if reports_dir.exists():
        reports = sorted(reports_dir.glob("retrain_report_*.json"))
        if reports:
            with open(reports[-1], "r", encoding="utf-8") as f:
                report = json.load(f)

    return {
        "status": "ok" if result.returncode == 0 else "error",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "report": report,
    }


@app.post("/reload_models")
def reload_models():
    """
    Перечитывает champion- и challenger-модели из model_registry.json.
    """
    global cat_fast, meta_vprd, cat_fast_challenger, meta_vprd_challenger
    global FAST_SHADOW_ENABLED, META_SHADOW_ENABLED

    try:
        model_registry_backend.load_champion()
        cat_fast = model_registry_backend.fast_gate
        meta_vprd = model_registry_backend.meta_brain

        cat_fast_challenger = model_registry_backend.fast_gate_challenger
        meta_vprd_challenger = model_registry_backend.meta_brain_challenger

        FAST_SHADOW_ENABLED = model_registry_backend.fast_gate_shadow_enabled
        META_SHADOW_ENABLED = model_registry_backend.meta_brain_shadow_enabled

        logger.info(
            "Reloaded models from registry: "
            "FastGate champion=%s, challenger=%s, shadow_enabled=%s; "
            "MetaBrain champion=%s, challenger=%s, shadow_enabled=%s",
            model_registry_backend.current_fast_version,
            model_registry_backend.challenger_fast_version,
            FAST_SHADOW_ENABLED,
            model_registry_backend.current_meta_version,
            model_registry_backend.challenger_meta_version,
            META_SHADOW_ENABLED,
        )

        return {
            "status": "ok",
            "fast_gate_version": model_registry_backend.current_fast_version,
            "meta_brain_version": model_registry_backend.current_meta_version,
            "fast_shadow_enabled": FAST_SHADOW_ENABLED,
            "meta_shadow_enabled": META_SHADOW_ENABLED,
        }
    except Exception as e:
        logger.exception("Ошибка при reload_models: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/promote_model")
def promote_model(brain: str):
    """
    Promote challenger -> champion для fast_gate или meta_brain.
    """
    if brain not in ("fast_gate", "meta_brain"):
        raise HTTPException(status_code=400, detail="brain должен быть 'fast_gate' или 'meta_brain'")

    global cat_fast, meta_vprd, cat_fast_challenger, meta_vprd_challenger
    global FAST_SHADOW_ENABLED, META_SHADOW_ENABLED

    try:
        model_registry_backend.promote(brain)
        # сразу перезагружаем champion-/challenger-модели
        model_registry_backend.load_champion()
        cat_fast = model_registry_backend.fast_gate
        meta_vprd = model_registry_backend.meta_brain
        cat_fast_challenger = model_registry_backend.fast_gate_challenger
        meta_vprd_challenger = model_registry_backend.meta_brain_challenger
        FAST_SHADOW_ENABLED = model_registry_backend.fast_gate_shadow_enabled
        META_SHADOW_ENABLED = model_registry_backend.meta_brain_shadow_enabled

        logger.info(
            "Promote completed for %s. Now FastGate champion=%s, MetaBrain champion=%s",
            brain,
            model_registry_backend.current_fast_version,
            model_registry_backend.current_meta_version,
        )

        return {
            "status": "ok",
            "brain": brain,
            "fast_gate_version": model_registry_backend.current_fast_version,
            "meta_brain_version": model_registry_backend.current_meta_version,
            "fast_shadow_enabled": FAST_SHADOW_ENABLED,
            "meta_shadow_enabled": META_SHADOW_ENABLED,
        }
    except Exception as e:
        logger.exception("Ошибка при promote_model(%s): %s", brain, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rollback_model")
def rollback_model(brain: str):
    """
    Rollback: champion <-> previous.
    """
    if brain not in ("fast_gate", "meta_brain"):
        raise HTTPException(status_code=400, detail="brain должен быть 'fast_gate' или 'meta_brain'")

    global cat_fast, meta_vprd, cat_fast_challenger, meta_vprd_challenger
    global FAST_SHADOW_ENABLED, META_SHADOW_ENABLED

    try:
        model_registry_backend.rollback(brain)
        model_registry_backend.load_champion()
        cat_fast = model_registry_backend.fast_gate
        meta_vprd = model_registry_backend.meta_brain
        cat_fast_challenger = model_registry_backend.fast_gate_challenger
        meta_vprd_challenger = model_registry_backend.meta_brain_challenger
        FAST_SHADOW_ENABLED = model_registry_backend.fast_gate_shadow_enabled
        META_SHADOW_ENABLED = model_registry_backend.meta_brain_shadow_enabled

        logger.info(
            "Rollback completed for %s. Now FastGate champion=%s, MetaBrain champion=%s",
            brain,
            model_registry_backend.current_fast_version,
            model_registry_backend.current_meta_version,
        )

        return {
            "status": "ok",
            "brain": brain,
            "fast_gate_version": model_registry_backend.current_fast_version,
            "meta_brain_version": model_registry_backend.current_meta_version,
            "fast_shadow_enabled": FAST_SHADOW_ENABLED,
            "meta_shadow_enabled": META_SHADOW_ENABLED,
        }
    except Exception as e:
        logger.exception("Ошибка при rollback_model(%s): %s", brain, e)
        raise HTTPException(status_code=500, detail=str(e))
