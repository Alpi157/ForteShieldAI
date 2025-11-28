import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

# ================== BASIC CONFIG ==================

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="ForteShield AI — Analyst",
    page_icon="F",
    layout="wide",
)

# ---------- brand-driven styling (Forte) ----------

st.markdown(
    """
    <style>
    :root {
        --forte-magenta: #A31551;
        --forte-dark-magenta: #4A0221;
        --forte-sky: #61CFFF;
        --forte-coral: #FC9385;
        --forte-deep-blue: #072146;
        --forte-mid-blue: #043263;
        --forte-bg: #faf5f7;
        --forte-bg-alt: #f3edf2;
        --forte-ink: #260013;
    }

    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--forte-ink);
    }

    body {
        background: radial-gradient(circle at top left, #ffe5f1 0, #f4f7fb 40%, #f9f0f5 100%);
    }

    .block-container {
        padding-top: 3.2rem;
        padding-bottom: 2rem;
        max-width: 1300px;
    }

    /* header */
    .forte-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.6rem;
    }

    .forte-logo-word {
        font-size: 40px;
        font-weight: 800;
        letter-spacing: 0.04em;
        color: var(--forte-magenta);
    }

    .forte-logo-dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        background: var(--forte-sky);
        display: inline-block;
        margin-left: 6px;
    }

    .forte-product-meta {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 0.15rem;
    }

    .forte-badge {
        font-size: 11px;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--forte-dark-magenta);
        background: rgba(163,21,81,0.07);
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        border: 1px solid rgba(163,21,81,0.18);
    }

    .forte-subtitle {
        font-size: 13px;
        color: #5b3a4d;
    }

    /* sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #220013 0, #04152f 70%);
    }

    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: #fef9fb;
    }

    .sidebar-caption {
        font-size: 12px;
        opacity: 0.8;
        margin-bottom: 0.7rem;
        color: #fef9fb;
    }

    .sidebar-card {
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(255,255,255,0.10);
        color: #fef9fb;
    }

    .sidebar-metric-label {
        font-size: 12px;
        opacity: 0.85;
    }
    .sidebar-metric-value {
        font-size: 16px;
        font-weight: 600;
        margin-top: 0.05rem;
    }

    .sidebar-section-label {
        font-size: 15px;
        font-weight: 700;
        color: #fef9fb;
        margin-top: 1.0rem;
        margin-bottom: 0.25rem;
        opacity: 0.95;
    }

    /* metrics */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(163,21,81,0.06), rgba(97,207,255,0.07));
        padding: 0.9rem 0.9rem;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(7,33,70,0.12);
        border: 1px solid rgba(255,255,255,0.9);
        backdrop-filter: blur(8px);
    }

    div[data-testid="stMetric"] > label {
        font-size: 13px;
        color: #4b1e35;
    }

    /* tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.9rem;
        margin-top: 0.1rem;
        margin-bottom: 0.7rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 15px;
        font-weight: 600;
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
        padding-left: 0.9rem;
        padding-right: 0.9rem;
        border-radius: 999px;
        margin-right: 0.1rem;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(163,21,81,0.06);
    }

    /* section cards */
    .forte-section-card {
        background: rgba(255,255,255,0.9);
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        border: 1px solid rgba(7,33,70,0.05);
        box-shadow: 0 12px 32px rgba(7,33,70,0.08);
        margin-bottom: 1rem;
    }

    .forte-section-title {
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 0.2rem;
    }

    .forte-pills {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        flex-wrap: wrap;
        margin-top: 0.35rem;
    }
    .forte-pill {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        background: rgba(7,33,70,0.03);
        border: 1px solid rgba(7,33,70,0.12);
        color: #283046;
    }

    /* tables */
    .dataframe td, .dataframe th {
        font-size: 13px !important;
    }

    /* buttons — общий стиль */
    div.stButton > button {
        border-radius: 999px;
        border: none;
        font-weight: 600;
        font-size: 13px;
        padding: 0.4rem 1.0rem;
        background: linear-gradient(135deg, #A31551, #FC9385);
        color: white;
        box-shadow: 0 8px 20px rgba(163,21,81,0.35);
    }
    div.stButton > button:hover {
        filter: brightness(1.05);
        box-shadow: 0 10px 24px rgba(7,33,70,0.35);
    }

    /* promote / rollback — чуть более «secondary» */
    .model-ops div.stButton > button {
        background: linear-gradient(135deg, #043263, #061e40);
        box-shadow: 0 8px 20px rgba(7,33,70,0.4);
    }
    .model-ops div.stButton > button:nth-child(2n) {
        background: linear-gradient(135deg, #A31551, #FC9385);
    }

    /* sidebar collapse / expand button — белая стрелка */
    button[title="Collapse sidebar"] svg,
    button[title="Expand sidebar"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="forte-header">
        <div>
            <div class="forte-logo-word">forte<span class="forte-logo-dot"></span></div>
        </div>
        <div class="forte-product-meta">
            <div class="forte-badge">ForteShield AI · Analyst console</div>
            <div class="forte-subtitle">
                Мониторинг антифрода в реальном времени: модели, кейсы и LLM-ассистент.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# data / SHAP paths
# ==================================================


def find_processed_dir() -> Path:
    here = Path(__file__).resolve().parent
    candidates_roots = [here, here.parent, Path.cwd()]
    for root in candidates_roots:
        p = root / "data" / "processed"
        if p.exists():
            return p
    return here / "data" / "processed"


PROCESSED_DIR = find_processed_dir()

SHAP_FILES = {
    "Fast Gate v11": [
        "fast_gate_shap_summary_v11.parquet",
        "shap_fast_gate_summary_v11.parquet",
    ],
    "Graph Brain v11": [
        "graph_brain_shap_summary_v11.parquet",
        "shap_graph_brain_summary_v11.parquet",
    ],
    "Session Brain v11": [
        "session_brain_shap_summary_v11.parquet",
        "shap_session_brain_summary_v11.parquet",
    ],
    "AE Brain meta v11": [
        "ae_meta_shap_summary_v11.parquet",
        "shap_ae_meta_summary_v11.parquet",
    ],
    "Sequence Brain meta v11": [
        "seq_meta_shap_summary_v11.parquet",
        "shap_seq_meta_summary_v11.parquet",
    ],
    "Meta Brain vProd": [
        "meta_vprod_shap_summary_v11.parquet",
        "shap_meta_vprod_summary_v11.parquet",
    ],
}

# период окна дашборда — последние N дней от max(created_at)
DASHBOARD_WINDOW_DAYS = int(os.getenv("DASHBOARD_WINDOW_DAYS", "1"))
DASHBOARD_CASE_LIMIT = int(os.getenv("DASHBOARD_CASE_LIMIT", "100000"))

# ==================================================
# helpers
# ==================================================


def safe_rerun():
    """
    Обёртка над st.rerun / st.experimental_rerun для совместимости версий.
    """
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


# ==================================================
# HTTP helpers
# ==================================================


def api_get(path: str, params=None):
    url = f"{BACKEND_URL}{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def api_post(path: str, json=None, params=None):
    url = f"{BACKEND_URL}{path}"
    r = requests.post(url, json=json, params=params, timeout=600)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=5, show_spinner=False)
def backend_health() -> dict:
    """
    Обёртка над /health. Возвращаем полный dict, чтобы знать и shadow-статусы.
    """
    try:
        return api_get("/health")
    except Exception:
        return {"status": "error"}


# ==================================================
# data from backend (cached where это не real-time)
# ==================================================


@st.cache_data(ttl=5, show_spinner=False)
def load_cases(limit: int = 500) -> pd.DataFrame:
    """
    Подтягиваем кейсы из backend (/cases).
    ttl=5 + periodic rerun дают примерно живую картинку.
    """
    try:
        data = api_get("/cases", params={"limit": limit})
    except Exception as e:
        st.error(f"Ошибка при запросе /cases: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if df.empty:
        return df

    # нормализуем даты
    for col in ["created_at", "transdatetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # унифицируем итоговый риск / решение
    if "risk_score_final" not in df.columns and "risk_meta" in df.columns:
        df["risk_score_final"] = df["risk_meta"]
    if "decision" not in df.columns:
        df["decision"] = "unknown"
    if "risk_type" not in df.columns:
        df["risk_type"] = "unknown"

    return df


@st.cache_data(ttl=20, show_spinner=False)
def load_strategy():
    try:
        return api_get("/strategy")
    except Exception:
        return None


@st.cache_data(ttl=20, show_spinner=False)
def load_strategies():
    try:
        return api_get("/strategies")
    except Exception:
        return []


def set_strategy(name: str):
    try:
        return api_post(f"/strategy/{name}")
    except Exception as e:
        st.error(f"Не удалось сменить стратегию: {e}")
        return None


@st.cache_data(ttl=20, show_spinner=False)
def load_model_registry():
    try:
        return api_get("/model_registry")
    except Exception as e:
        st.error(f"Ошибка при запросе /model_registry: {e}")
        return None


@st.cache_data(ttl=30, show_spinner=False)
def load_last_retrain_report():
    try:
        return api_get("/retrain_last_report")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_shap_summary(possible_filenames):
    if isinstance(possible_filenames, str):
        possible_filenames = [possible_filenames]

    chosen_path = None
    df = None

    for fname in possible_filenames:
        path = PROCESSED_DIR / fname
        if path.exists():
            chosen_path = path
            df = pd.read_parquet(path)
            break

    if df is None or df.empty:
        return None, PROCESSED_DIR / possible_filenames[0]

    feature_col = None
    for cand in ["feature", "feature_name", "col", "column"]:
        if cand in df.columns:
            feature_col = cand
            break
    if feature_col is None:
        df = df.reset_index()
        feature_col = df.columns[0]

    numeric_cols = [
        c for c in df.columns
        if c != feature_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        return None, chosen_path

    value_col = numeric_cols[0]
    out = (
        df[[feature_col, value_col]]
        .rename(columns={feature_col: "feature", value_col: "mean_abs_shap"})
        .sort_values("mean_abs_shap", ascending=False)
        .head(25)
    )
    return out, chosen_path


# ==================================================
# derived metrics for dashboard
# ==================================================


def filter_recent_cases(df: pd.DataFrame, days: int = DASHBOARD_WINDOW_DAYS) -> pd.DataFrame:
    """
    Берём только последние N дней относительно максимального created_at / transdatetime.
    Это отрезает старый оффлайн-датасет и оставляет то, что реально происходит сейчас.
    """
    if df.empty:
        return df

    ts_col = "created_at" if "created_at" in df.columns else "transdatetime"
    if ts_col not in df.columns:
        return df

    ts = pd.to_datetime(df[ts_col], errors="coerce")
    if ts.isna().all():
        return df

    latest = ts.max()
    cutoff = latest - pd.Timedelta(days=days)
    mask = ts >= cutoff
    return df.loc[mask].copy()


def compute_alert_metrics(df: pd.DataFrame):
    """
    Считает метрики по уже отфильтрованному фрейму (обычно по последним дням).
    """
    if df.empty:
        return 0, 0, 0.0

    total = len(df)
    alerts = int((df["decision"] == "alert").sum()) if "decision" in df.columns else 0
    rate = (alerts / total * 100.0) if total > 0 else 0.0
    return int(total), int(alerts), float(rate)


def build_alert_ts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    ts_df = df.copy()
    ts_col = "created_at" if "created_at" in ts_df.columns else "transdatetime"
    ts_df[ts_col] = pd.to_datetime(ts_df[ts_col], errors="coerce")
    ts_df = ts_df.dropna(subset=[ts_col])
    ts_df["date"] = ts_df[ts_col].dt.date

    all_per_day = ts_df.groupby("date")["id"].count().rename("total_cases")

    if "decision" in ts_df.columns:
        alerts_per_day = (
            ts_df[ts_df["decision"] == "alert"]
            .groupby("date")["id"]
            .count()
            .rename("alerts")
        )
    else:
        alerts_per_day = pd.Series(dtype=int, name="alerts")

    out = pd.concat([all_per_day, alerts_per_day], axis=1).fillna(0).reset_index()
    return out


def build_risk_type_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "risk_type" not in df.columns:
        return pd.DataFrame()
    tmp = (
        df["risk_type"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("risk_type")
        .reset_index(name="count")
    )
    return tmp


def build_risk_score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "risk_score_final" not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp = tmp.dropna(subset=["risk_score_final"])
    if "decision" not in tmp.columns:
        tmp["decision"] = "unknown"
    return tmp


def shap_chart(df: pd.DataFrame, title: str):
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("mean_abs_shap:Q", title="Средний |SHAP|"),
            y=alt.Y("feature:N", sort="-x", title="Фича"),
            color=alt.value("#A31551"),
            tooltip=["feature", "mean_abs_shap"],
        )
        .properties(title=title, height=380)
    )
    return chart


# ==================================================
# helpers for Model Governance (champion vs challenger)
# ==================================================


def extract_versions_from_registry(brain_cfg):
    """
    Пытаемся вытащить подписи для champion / challenger из model_registry.
    """
    if not isinstance(brain_cfg, dict):
        return None, None

    champ = (
        brain_cfg.get("champion")
        or brain_cfg.get("champion_model")
        or brain_cfg.get("current")
        or {}
    )
    chall = (
        brain_cfg.get("challenger")
        or brain_cfg.get("challenger_model")
        or brain_cfg.get("candidate")
        or {}
    )

    def label(obj):
        if not isinstance(obj, dict):
            return "—"
        for key in ["version", "name", "model_id", "tag"]:
            if key in obj:
                return str(key and obj[key])
        if "path" in obj:
            try:
                return Path(obj["path"]).name
            except Exception:
                return str(obj["path"])
        return "—"

    return label(champ), label(chall)


def extract_brain_comparison(report: dict, brain_key: str):
    """
    Парсим retrain_report, вытаскиваем champion vs challenger для конкретного мозга.
    Поддерживаем несколько возможных вариантов именования пар: champion/challenger,
    before/after, old/new, baseline/candidate, current/retrained.
    """
    if not isinstance(report, dict):
        return None

    section = report.get(brain_key)
    if not isinstance(section, dict):
        return None

    candidates = [
        ("champion", "challenger"),
        ("before", "after"),
        ("old", "new"),
        ("baseline", "candidate"),
        ("current", "retrained"),
    ]

    pair = None
    for a, b in candidates:
        if a in section and b in section:
            pair = (a, b)
            break

    if pair is None:
        return None

    key_champ, key_chall = pair
    champ = section.get(key_champ, {})
    chall = section.get(key_chall, {})

    def get_metrics(obj):
        if not isinstance(obj, dict):
            return {}
        if isinstance(obj.get("metrics"), dict):
            return obj["metrics"]
        if isinstance(obj.get("eval"), dict):
            return obj["eval"]
        metrics = {}
        for k, v in obj.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metrics[k] = v
        return metrics

    m_champ = get_metrics(champ)
    m_chall = get_metrics(chall)

    keys = sorted(set(m_champ.keys()) | set(m_chall.keys()))
    if not keys:
        return None

    rows = []
    for k in keys:
        rows.append(
            {
                "metric": k,
                "champion": m_champ.get(k),
                "challenger": m_chall.get(k),
            }
        )
    df = pd.DataFrame(rows)

    return {
        "df": df,
        "champion_obj": champ,
        "challenger_obj": chall,
        "label_champion": key_champ,
        "label_challenger": key_chall,
    }


def comparison_chart(df: pd.DataFrame, title: str):
    long = df.melt(
        id_vars=["metric"],
        value_vars=["champion", "challenger"],
        var_name="model",
        value_name="value",
    )
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("metric:N", title="Метрика"),
            y=alt.Y("value:Q", title="Значение"),
            color=alt.Color(
                "model:N",
                title="",
                scale=alt.Scale(
                    domain=["champion", "challenger"],
                    range=["#043263", "#A31551"],
                ),
            ),
            tooltip=["metric:N", "model:N", "value:Q"],
        )
        .properties(title=title, height=260)
    )
    return chart


# ==================================================
# SIDEBAR (strategy, health)
# ==================================================

with st.sidebar:
    st.markdown('<div class="sidebar-title">ForteShield AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-caption">Онлайн-консоль аналитика: выбор стратегии, мониторинг моделей и LLM-ассистент.</div>',
        unsafe_allow_html=True,
    )

    # health
    health_info = backend_health()
    status = health_info.get("status", "unknown")
    color = {"ok": "#52d273", "error": "#ff5c5c"}.get(status, "#ffb020")
    status_label = {"ok": "Backend online", "error": "Backend недоступен"}.get(
        status, "Статус неизвестен"
    )
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-metric-label">Состояние backend</div>
            <div class="sidebar-metric-value" style="display:flex;align-items:center;gap:0.4rem;">
                <span style="width:10px;height:10px;border-radius:999px;background:{color};display:inline-block;"></span>
                <span>{status_label}</span>
            </div>
            <div style="font-size:11px;margin-top:0.3rem;opacity:0.8;">
                Shadow Fast Gate: <b>{"on" if health_info.get("fast_shadow_enabled") else "off"}</b><br/>
                Shadow Meta Brain: <b>{"on" if health_info.get("meta_shadow_enabled") else "off"}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # current strategy info
    current_strategy = load_strategy()
    current_name = current_strategy.get("name") if isinstance(current_strategy, dict) else None
    threshold = (
        current_strategy.get("config", {}).get("threshold", "—")
        if isinstance(current_strategy, dict)
        else "—"
    )

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-metric-label">Текущая стратегия</div>
            <div class="sidebar-metric-value">{current_name or "неизвестно"}</div>
            <div style="font-size:12px;margin-top:0.15rem;opacity:0.8;">
                Порог алерта: <b>{threshold}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # strategy selector
    st.markdown(
        '<div class="sidebar-section-label">Режим принятия решений</div>',
        unsafe_allow_html=True,
    )

    raw_strats = load_strategies()
    strat_names: list[str] = []

    if isinstance(raw_strats, list):
        for s in raw_strats:
            if isinstance(s, dict):
                name = s.get("name") or s.get("id") or s.get("strategy") or str(s)
            else:
                name = str(s)
            strat_names.append(name)
    elif isinstance(raw_strats, dict):
        if "strategies" in raw_strats:
            s_block = raw_strats["strategies"]
            if isinstance(s_block, list):
                for s in s_block:
                    if isinstance(s, dict):
                        name = s.get("name") or s.get("id") or s.get("strategy") or str(s)
                    else:
                        name = str(s)
                    strat_names.append(name)
            elif isinstance(s_block, dict):
                strat_names = list(s_block.keys())
        else:
            for k in raw_strats.keys():
                if k not in ("current", "default", "active"):
                    strat_names.append(str(k))

    strat_names = sorted(set(strat_names))

    if strat_names:
        default_index = 0
        if current_name and current_name in strat_names:
            default_index = strat_names.index(current_name)
        selected = st.selectbox(
            "",
            options=strat_names,
            index=default_index,
            key="strategy_select",
        )
        if st.button("Применить стратегию", use_container_width=True):
            res = set_strategy(selected)
            if res:
                st.success(f"Стратегия переключена на {selected}")
                load_strategy.clear()
                load_cases.clear()
                safe_rerun()
    else:
        st.caption("Стратегии недоступны (проверь endpoint `/strategies`).")

# ==================================================
# TABS
# ==================================================

(
    tab_dashboard,
    tab_cases,
    tab_llm,
    tab_shap_tab,
    tab_governance,
) = st.tabs(
    [
        "Dashboard",
        "Cases",
        "LLM Assistant",
        "SHAP insights",
        "Model Governance",
    ]
)

# ---------------- TAB 1: DASHBOARD ----------------

def dashboard_core():
    st.markdown(
        f"""
        <div class="forte-section-card">
            <div class="forte-section-title">Сводка по последним кейсам</div>
            <div style="font-size:13px;opacity:0.85;">
                Ниже — живые метрики по последним транзакциям, прошедшим через
                <code>/score_transaction</code>. Окно: последние {DASHBOARD_WINDOW_DAYS} дней по данным backend.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Берём чуть больше кейсов, чтобы статистика по окну была честнее
    cases_df_all = load_cases(limit=DASHBOARD_CASE_LIMIT)
    cases_df = filter_recent_cases(cases_df_all)

    total_cases, alerts_count, alert_rate = compute_alert_metrics(cases_df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего кейсов (в окне)", total_cases)
    with col2:
        st.metric("Алертов (в окне)", alerts_count)
    with col3:
        st.metric("Alert rate в окне", f"{alert_rate:.1f} %")

    st.markdown("### Динамика алертов и профиль риска")

    col_ts, col_risk_type = st.columns([2.1, 1.2])

    with col_ts:
        ts_df = build_alert_ts(cases_df)
        if not ts_df.empty:
            ts_melt = ts_df.melt(
                id_vars=["date"],
                value_vars=["total_cases", "alerts"],
                var_name="metric",
                value_name="value",
            )
            chart_ts = (
                alt.Chart(ts_melt)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="Дата"),
                    y=alt.Y("value:Q", title="Количество кейсов"),
                    color=alt.Color(
                        "metric:N",
                        title="Метрика",
                        scale=alt.Scale(
                            domain=["total_cases", "alerts"],
                            range=["#043263", "#A31551"],
                        ),
                    ),
                    tooltip=["date:T", "metric:N", "value:Q"],
                )
                .properties(height=320)
            )
            st.altair_chart(chart_ts, use_container_width=True)
        else:
            st.info(
                "Нет данных по динамике кейсов в выбранном окне. "
                "Сначала прогоняй транзакции через backend."
            )

    with col_risk_type:
        rt_df = build_risk_type_df(cases_df)
        if not rt_df.empty:
            chart_rt = (
                alt.Chart(rt_df)
                .mark_bar()
                .encode(
                    x=alt.X("risk_type:N", title="Тип риска"),
                    y=alt.Y("count:Q", title="Количество кейсов"),
                    color=alt.Color(
                        "risk_type:N",
                        scale=alt.Scale(scheme="tableau20"),
                        legend=None,
                    ),
                    tooltip=["risk_type", "count"],
                )
                .properties(height=320)
            )
            st.altair_chart(chart_rt, use_container_width=True)
        else:
            st.info("Типы риска пока недоступны (все `unknown` или нет кейсов в окне).")

    st.markdown("### Распределение итогового риска (allow vs alert)")
    dist_df = build_risk_score_distribution(cases_df)
    if not dist_df.empty:
        chart_dist = (
            alt.Chart(dist_df)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X(
                    "risk_score_final:Q",
                    bin=alt.Bin(maxbins=30),
                    title="risk_score_final",
                ),
                y=alt.Y("count():Q", title="Количество кейсов"),
                color=alt.Color(
                    "decision:N",
                    title="Решение",
                    scale=alt.Scale(
                        domain=["allow", "alert", "unknown"],
                        range=["#61CFFF", "#A31551", "#b0b0b0"],
                    ),
                ),
                tooltip=["decision", "count():Q"],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(chart_dist, use_container_width=True)
    else:
        st.info("Нет данных для распределения risk_score_final в выбранном окне.")


# автообновление дашборда каждые 5 секунд, если в версии Streamlit есть st.fragment
if hasattr(st, "fragment"):
    @st.fragment(run_every="5s")
    def dashboard_runner():
        dashboard_core()
else:
    def dashboard_runner():
        dashboard_core()

with tab_dashboard:
    dashboard_runner()

# ---------------- TAB 2: CASES ----------------

def cases_core():
    st.markdown(
        """
        <div class="forte-section-card">
            <div class="forte-section-title">Очередь кейсов</div>
            <div style="font-size:13px;opacity:0.85;">
                Здесь можно просмотреть транзакции, отфильтровать их по решению и риску,
                провалиться в детали и проставить разметку для retrain.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_header_left, col_header_right = st.columns([1.0, 0.35])
    with col_header_left:
        st.markdown("#### Список кейсов")
    with col_header_right:
        if st.button("Обновить данные", use_container_width=True, key="btn_refresh_cases"):
            load_cases.clear()
            safe_rerun()

    cases_df = load_cases(limit=500)

    if cases_df.empty:
        st.info("Кейсов пока нет. Сначала прогоняй транзакции через endpoint `/score_transaction`.")
        return

    with st.expander("Фильтры", expanded=True):
        col_f1, col_f2, col_f3 = st.columns([1.1, 2.0, 1.6])

        with col_f1:
            decision_filter = st.multiselect(
                "Решение",
                options=["allow", "alert", "unknown"],
                default=["allow", "alert", "unknown"],
                key="cases_decision_filter",
            )

        with col_f2:
            min_risk = float(cases_df["risk_score_final"].min())
            max_risk = float(cases_df["risk_score_final"].max())
            risk_range = st.slider(
                "Диапазон итогового риска",
                min_value=float(round(min_risk, 3)),
                max_value=float(round(max_risk, 3)),
                value=(float(round(min_risk, 3)), float(round(max_risk, 3))),
                key="cases_risk_range",
            )

        with col_f3:
            search = st.text_input("Поиск по docno / client_id / label", key="cases_search")

    view_df = cases_df.copy()
    if decision_filter:
        view_df = view_df[view_df["decision"].isin(decision_filter)]
    view_df = view_df[
        (view_df["risk_score_final"] >= risk_range[0])
        & (view_df["risk_score_final"] <= risk_range[1])
    ]
    if search:
        search_lower = search.lower()
        mask = (
            view_df["docno"].astype(str).str.contains(search_lower)
            | view_df["cst_dim_id"].astype(str).str.contains(search_lower)
            | view_df.get("user_label", "").astype(str).str.contains(search_lower)
        )
        view_df = view_df[mask]

    cols_show = [
        "id",
        "docno",
        "cst_dim_id",
        "direction",
        "amount",
        "transdatetime",
        "risk_score_final",
        "decision",
        "risk_type",
    ]
    if "user_label" in view_df.columns:
        cols_show.append("user_label")
    if "use_for_retrain" in view_df.columns:
        cols_show.append("use_for_retrain")

    st.dataframe(
        view_df[cols_show],
        use_container_width=True,
        hide_index=True,
    )

    case_ids = view_df["id"].tolist()
    if not case_ids:
        st.info("Нет кейсов после применения фильтров.")
        return

    selected_id = st.selectbox("Выбери кейс для деталей", case_ids, key="cases_selected_id")
    if not selected_id:
        return

    case = api_get(f"/case/{selected_id}")

    col_a, col_b = st.columns([1.1, 1.4])
    with col_a:
        st.markdown("#### Карточка кейса")
        risk_val = case.get("risk_meta") or case.get("risk_score_final")
        if risk_val is not None:
            st.metric("Итоговый риск (meta)", round(float(risk_val), 4))
        st.metric("Сумма", f"{case.get('amount')} ₸")
        st.metric("Решение", case.get("decision"))
        st.metric("Тип риска", case.get("risk_type") or "unknown")

        # champion vs shadow (если есть)
        rf = case.get("risk_fast")
        rf_sh = case.get("risk_fast_shadow")
        rm = case.get("risk_meta")
        rm_sh = case.get("risk_meta_shadow")

        lines = []
        if rf is not None and rf_sh is not None:
            lines.append(
                f"Fast Gate: champion={round(float(rf), 4)} · shadow={round(float(rf_sh), 4)}"
            )
        if rm is not None and rm_sh is not None:
            lines.append(
                f"Meta Brain: champion={round(float(rm), 4)} · shadow={round(float(rm_sh), 4)}"
            )
        if lines:
            st.markdown(
                "**Shadow-скоринг challenger (только в логах, без влияния на решение):**  \n"
                + "  \n".join(lines)
            )

    with col_b:
        st.markdown("#### Детали (сырые поля)")
        anomaly_val = case.get("anomaly_score_emb")
        case_display = dict(case)
        case_display.pop("anomaly_score_emb", None)
        st.json(case_display)
        # отдельная строка для anomaly_score_emb, чтобы не видеть сырое null
        if anomaly_val is None:
            st.caption("anomaly_score_emb: n/a (embedding-анализ отключён в этой демо)")
        else:
            try:
                v = float(anomaly_val)
                st.caption(f"anomaly_score_emb: {v:.6f}")
            except Exception:
                st.caption(f"anomaly_score_emb: {anomaly_val}")

    # ------- разметка аналитика -------
    st.markdown("#### Разметка аналитика и retrain")

    col_lab1, col_lab2 = st.columns([1.2, 1.4])

    with col_lab1:
        cur_label = case.get("user_label")
        raw_use = case.get("use_for_retrain")
        use_flag = True if raw_use is None else bool(raw_use)
        cur_status = case.get("status") or "new"

        label_map_display = {
            "fraud": "Fraud",
            "legit": "Legit",
            "unknown": "Не задано",
            None: "Не задано",
        }
        label_display = label_map_display.get(cur_label, "Не задано")

        st.markdown(
            f"""
            **Текущая разметка:** {label_display}  
            **Use for retrain:** {"Да" if use_flag else "Нет"}  
            **Статус:** {cur_status}
            """
        )

    with col_lab2:
        label_options = [("Не задано", None), ("Fraud", "fraud"), ("Legit", "legit")]
        current_label = case.get("user_label")
        if current_label in ("fraud", "Fraud"):
            default_idx = 1
        elif current_label in ("legit", "Legit"):
            default_idx = 2
        else:
            default_idx = 0

        label_names = [x[0] for x in label_options]
        label_choice = st.radio(
            "Label аналитика",
            label_names,
            index=default_idx,
            key=f"lbl_{selected_id}",
        )
        label_value = dict(label_options)[label_choice]

        raw_use = case.get("use_for_retrain")
        use_default = True if raw_use is None else bool(raw_use)
        use_for_retrain_choice = st.checkbox(
            "Использовать кейс при retrain моделей",
            value=use_default,
            key=f"use_retrain_{selected_id}",
        )

        status_options = ["new", "in_review", "closed"]
        cur_status = case.get("status") or "new"
        if cur_status not in status_options:
            status_options.append(cur_status)
        status_index = status_options.index(cur_status)
        status_choice = st.selectbox(
            "Статус кейса",
            status_options,
            index=status_index,
            key=f"status_{selected_id}",
        )

        if st.button("Сохранить разметку", key=f"save_label_{selected_id}"):
            payload = {
                "user_label": label_value,
                "use_for_retrain": use_for_retrain_choice,
                "status": status_choice,
            }
            try:
                _ = api_post(f"/case/{selected_id}/label", json=payload)
                st.success("Разметка сохранена.")
                load_cases.clear()
                safe_rerun()
            except Exception as e:
                st.error(f"Ошибка при обновлении разметки: {e}")


# автообновление Cases каждые 5 секунд (если есть st.fragment)
if hasattr(st, "fragment"):
    @st.fragment(run_every="5s")
    def cases_runner():
        cases_core()
else:
    def cases_runner():
        cases_core()

with tab_cases:
    cases_runner()

# ---------------- TAB 3: LLM Assistant ----------------
with tab_llm:
    st.markdown(
        """
        <div class="forte-section-card">
            <div class="forte-section-title">LLM-ассистент по кейсам</div>
            <div style="font-size:13px;opacity:0.85;">
                Выбери кейс, и ассистент объяснит решение моделей человеческим языком
                или подготовит черновик SAR-отчёта.
            </div>
            <div class="forte-pills">
                <span class="forte-pill">Meta Brain</span>
                <span class="forte-pill">SHAP</span>
                <span class="forte-pill">Forte compliance</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cases_df = load_cases(limit=200)
    if cases_df.empty:
        st.info("Нет кейсов для объяснения. Сначала прогоняй транзакции через backend.")
    else:
        cases_df = cases_df.sort_values(
            by="transdatetime" if "transdatetime" in cases_df.columns else "created_at",
            ascending=False,
        )

        col_sel1, col_sel2 = st.columns([2.0, 1.0])
        with col_sel1:
            search_llm = st.text_input("Поиск по docno / client_id для LLM", "", key="llm_search")
        filt_df = cases_df.copy()
        if search_llm:
            mask = (
                filt_df["docno"].astype(str).str.contains(search_llm)
                | filt_df["cst_dim_id"].astype(str).str.contains(search_llm)
            )
            filt_df = filt_df[mask]

        def label_row(r):
            return (
                f"#{r['id']} • docno {r['docno']} • клиент {r['cst_dim_id']} • "
                f"{r['amount']} ₸ • {r['decision']}"
            )

        if not filt_df.empty:
            filt_df["label"] = filt_df.apply(label_row, axis=1)
            labels = filt_df["label"].tolist()
            id_by_label = dict(zip(filt_df["label"], filt_df["id"]))
        else:
            labels = []
            id_by_label = {}

        with col_sel2:
            st.write("")
            st.write("")

        if not labels:
            st.info("По фильтру не найдено кейсов.")
        else:
            selected_label = st.selectbox("Кейс для объяснения / SAR", labels, key="llm_case_select")
            selected_id = id_by_label[selected_label]

            case = api_get(f"/case/{selected_id}")
            st.markdown("#### Краткая сводка по кейсу")
            col_l, col_r = st.columns(2)
            with col_l:
                st.metric("Сумма", f"{case.get('amount')} ₸")
                risk_val = case.get("risk_meta") or case.get("risk_score_final")
                if risk_val is not None:
                    st.metric("Итоговый риск", round(float(risk_val), 4))
            with col_r:
                st.metric("Решение", case.get("decision"))
                st.metric("Тип риска", case.get("risk_type") or "unknown")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Объяснить решение (LLM)", key="btn_explain"):
                    try:
                        resp = api_post("/llm/explain_case", params={"case_id": selected_id})
                        st.markdown("##### Объяснение для аналитика")
                        st.write(resp.get("explanation", "Пустой ответ от LLM."))
                    except Exception as e:
                        st.error(f"Ошибка при вызове /llm/explain_case: {e}")

            with col2:
                if st.button("Сгенерировать черновик SAR", key="btn_sar"):
                    try:
                        resp = api_post(
                            "/llm/generate_sar", params={"case_id": selected_id}
                        )
                        st.markdown("##### Черновик SAR")
                        st.write(resp.get("sar", "Пустой ответ от LLM."))
                    except Exception as e:
                        st.error(f"Ошибка при вызове /llm/generate_sar: {e}")

# ---------------- TAB 4: SHAP ----------------
with tab_shap_tab:
    st.markdown(
        f"""
        <div class="forte-section-card">
            <div class="forte-section-title">Глобальные важности признаков (SHAP)</div>
            <div style="font-size:13px;opacity:0.85;">
                Здесь можно увидеть, какие признаки в среднем сильнее всего влияют на решения
                разных мозгов. Файлы читаются из каталога
                <code>{PROCESSED_DIR}</code>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    i = 0
    for model_name, filenames in SHAP_FILES.items():
        df_shap, path_used = load_shap_summary(filenames)
        col = cols[i % 2]
        with col:
            st.markdown(f"#### {model_name}")
            if df_shap is None or df_shap.empty:
                st.info(
                    "Не найден файл SHAP или он пуст.\n\n"
                    "Ожидаемый путь (один из):\n"
                    + "\n".join(f"- `{PROCESSED_DIR / f}`" for f in filenames)
                )
            else:
                ch = shap_chart(df_shap, model_name)
                st.altair_chart(ch, use_container_width=True)
        i += 1

# ---------------- TAB 5: Model Governance ----------------
with tab_governance:
    st.markdown(
        """
        <div class="forte-section-card">
            <div class="forte-section-title">Model Governance · champion / challenger</div>
            <div style="font-size:13px;opacity:0.85;">
                Управление версиями Fast Gate и Meta Brain, запуск перетренировки и визуальное
                сравнение качества champion vs challenger по последнему retrain.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([2, 1.2])

    # левая часть: версии и сравнение метрик
    with col_l:
        registry = load_model_registry()
        st.markdown("### Версии моделей (из model_registry)")
        if registry is None:
            st.info("model_registry.json пока недоступен.")
        else:
            fg_cfg = registry.get("fast_gate", {})
            mb_cfg = registry.get("meta_brain", {})

            fg_champ, fg_chall = extract_versions_from_registry(fg_cfg)
            mb_champ, mb_chall = extract_versions_from_registry(mb_cfg)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Fast Gate")
                st.markdown(
                    f"""
                    **Champion:** {fg_champ or "—"}  
                    **Challenger:** {fg_chall or "—"}
                    """
                )
            with c2:
                st.markdown("#### Meta Brain")
                st.markdown(
                    f"""
                    **Champion:** {mb_champ or "—"}  
                    **Challenger:** {mb_chall or "—"}
                    """
                )

            with st.expander("Показать сырой model_registry.json"):
                st.json(registry)

        st.markdown("### Сравнение качества по последнему retrain_report")

        report = load_last_retrain_report()
        if report is None:
            st.info(
                "Отчёт retrain_report_* пока не найден. Сначала запусти перетренировку через кнопку справа."
            )
        else:
            fg_comp = extract_brain_comparison(report, "fast_gate")
            mb_comp = extract_brain_comparison(report, "meta_brain")

            if fg_comp is None and mb_comp is None:
                st.info(
                    "Не удалось автоматически распарсить сравнение champion vs challenger из отчёта. "
                    "Посмотри структуру отчёта в raw JSON ниже."
                )
            else:
                if fg_comp is not None:
                    st.markdown("#### Fast Gate")
                    df = fg_comp["df"]
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    ch = comparison_chart(df, "Fast Gate: champion vs challenger")
                    st.altair_chart(ch, use_container_width=True)

                if mb_comp is not None:
                    st.markdown("#### Meta Brain")
                    df = mb_comp["df"]
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    ch = comparison_chart(df, "Meta Brain: champion vs challenger")
                    st.altair_chart(ch, use_container_width=True)

            with st.expander("Показать сырой retrain_report_* .json"):
                st.json(report)

    # правая часть — операции с моделями
    with col_r:
        st.markdown('<div class="model-ops">', unsafe_allow_html=True)
        st.markdown("### Операции с моделями")

        if st.button("Перетренировать challenger (Fast Gate + Meta)", key="btn_retrain"):
            with st.spinner("Запуск retrain_models.py..."):
                try:
                    resp = api_post("/retrain")
                    st.success(
                        f"retrain_models.py завершился, код={resp.get('returncode')}"
                    )
                    if resp.get("stderr_tail"):
                        st.text_area(
                            "stderr (последние строки)",
                            resp["stderr_tail"],
                            height=150,
                        )
                    load_model_registry.clear()
                    load_last_retrain_report.clear()
                except Exception as e:
                    st.error(f"Ошибка при /retrain: {e}")

        st.markdown("---")
        st.markdown("#### Promote / Rollback")

        if st.button("Promote Fast Gate challenger → champion", key="btn_promote_fast"):
            try:
                resp = api_post("/promote_model", params={"brain": "fast_gate"})
                st.success(f"Fast Gate: {resp}")
                api_post("/reload_models")
                load_model_registry.clear()
            except Exception as e:
                st.error(f"Ошибка при /promote_model fast_gate: {e}")

        if st.button("Promote Meta Brain challenger → champion", key="btn_promote_meta"):
            try:
                resp = api_post("/promote_model", params={"brain": "meta_brain"})
                st.success(f"Meta Brain: {resp}")
                api_post("/reload_models")
                load_model_registry.clear()
            except Exception as e:
                st.error(f"Ошибка при /promote_model meta_brain: {e}")

        st.markdown("—")

        if st.button("Rollback Fast Gate (champion ↔ previous)", key="btn_rb_fast"):
            try:
                resp = api_post("/rollback_model", params={"brain": "fast_gate"})
                st.success(f"Fast Gate rollback: {resp}")
                api_post("/reload_models")
                load_model_registry.clear()
            except Exception as e:
                st.error(f"Ошибка при /rollback_model fast_gate: {e}")

        if st.button("Rollback Meta Brain (champion ↔ previous)", key="btn_rb_meta"):
            try:
                resp = api_post("/rollback_model", params={"brain": "meta_brain"})
                st.success(f"Meta Brain rollback: {resp}")
                api_post("/reload_models")
                load_model_registry.clear()
            except Exception as e:
                st.error(f"Ошибка при /rollback_model meta_brain: {e}")

        st.markdown("</div>", unsafe_allow_html=True)
