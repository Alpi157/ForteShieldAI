# ForteShield AI: Multi-Layer Anti-Fraud Platform for ForteBank

**ForteShield AI** is an end-to-end solution that covers the full path from raw data to the analyst’s workstation:
from offline training of model ensembles to real-time transaction scoring, dashboards for the anti-fraud team, and a model governance loop with retrain, champion or challenger setup, and an LLM assistant.

---

## Contents

- [Business context and use case](#business-context-and-use-case)
- [Solution architecture](#solution-architecture)
- [Repository structure](#repository-structure)
- [Online scoring and backend (FastAPI)](#online-scoring-and-backend-fastapi)
- [Offline Feature Store](#1-offline-feature-store)
- [Online Feature Store](#2-online-feature-store)
- [Scoring brains (Brains)](#3-scoring-brains)
- [PolicyEngine](#4-policy-engine)
- [CaseStore](#5-case-store)
- [ModelRegistryBackend and model governance](#6-model-registry-backend)
- [REST API](#7-rest-api)
- [Shadow mode](#8-shadow-mode)
- [Explanations and LLM assistant](#explanations-and-llm-assistant)
- [Analyst console (Streamlit UI)](#analyst-console-streamlit-ui)
- [Testing and simulation scenarios](#testing-and-simulation-scenarios)
- [Retrain pipeline and champion or challenger](#retrain-pipeline--championchallenger)
- [Key strengths of ForteShield AI](#key-strengths-of-forteshield-ai)
- [How to run locally (example)](#how-to-run-locally-example)
- [Summary](#summary)

---

## Business Context and Use Case

The initial use case is fraudulent transactions in ForteBank:
- Financial operations of customers (a classic transactional dataset)
- An additional dataset with behavioral or session-level data (logins, online activity, in-app behavior)

**ForteShield AI** addresses several practical needs:

### Fast Decision for Every Operation
For every incoming transaction, the system estimates risk in milliseconds and decides whether to:
- Let it pass
- Raise an alert
- Flag it as a high-value risk case ("whale")

### Deep Analytics on Cases
The analyst should be able to:
- See a prioritized queue of cases
- Understand why the model made a specific decision
- Use a convenient interface for labeling (fraud or legit) and retraining

### Model Management
The risk team expects:
- New model versions trained on fresh labeled data
- A clear champion/challenger setup
- Shadow mode and safe rollback

### Compliance Support and SAR
For complex cases:
- Text explanations in human-readable language
- SAR report drafts ready for regulatory adaptation

ForteShield AI covers this entire cycle within a single repository.

---

## Solution Architecture

At the top level, the project is split into several layers:

- `notebooks/` – Offline pipeline for feature creation and model training (v11 or vprod)
- `backend/` – FastAPI service for scoring and model governance
- `ui/` – Streamlit-based analyst console (dashboard, LLM assistant, model governance)
- `models/` – Model artifacts (CatBoost and sklearn pipelines)
- `config/` – Feature schemas, thresholds, model registry, feature dictionary
- `data/processed/` – Offline feature store and SHAP summaries
- `scripts/` – Utilities for stream simulation and red team scenarios
- `retrain_models.py` – Script for retraining challenger models


---

## Repository Structure

```
.
├── notebooks/           # Offline EDA, features, model training, SHAP
├── backend/             # FastAPI scoring service and model governance
│   ├── main.py          # Entry point, REST API
│   ├── feature_store.py # OnlineFeatureStore
│   ├── policy_engine.py # PolicyEngine with risk strategies
│   ├── case_store.py    # SQLite based case storage
│   ├── model_loader.py  # ModelRegistryBackend
│   ├── llm_agent.py     # LLM explanations and SAR drafts
│   └── ...
├── ui/
│   └── app.py           # Streamlit analyst console
├── models/              # CatBoost, sklearn pipelines, Meta Brain
├── config/
│   ├── features_schema.json
│   ├── meta_thresholds_vprod.json
│   ├── model_registry.json
│   ├── feature_dictionary.json
│   └── ...
├── data/
│   └── processed/
│       ├── features_offline_v11.parquet
│       └── shap_*.parquet
├── scripts/
│   ├── stream_simulator.py
│   └── red_team.py
└── retrain_models.py    # Retraining of challenger models
```


# Online Scoring and Backend (FastAPI)

The core service lives in `backend/main.py` and is run as a **FastAPI** application.

---

## 1. Offline Feature Store

The Offline Feature Store is the backbone of the scoring pipeline. Implemented as a single Parquet file:

```
data/processed/features_offline_v11.parquet
```

This file is loaded into memory at backend startup (`backend/main.py`). No need to access raw transactional tables at runtime.

### Key Properties

#### Indexed by `docno`

- One row per transaction
- Enables O(1) lookups for scoring
- If `docno` is missing → fallback baseline risk is applied, but case is still logged

#### Stable Feature Definitions

Models are trained against the same schema (`v11`):

- Fast Gate: `features_schema_v11.json`
- Graph Brain: `graph_brain_features_v11.json`
- Session Brain: `session_features_v11.json`
- AE/Sequence heads: `autoencoder_meta_features_v11.json`, `sequence_meta_features_v11.json`
- Meta Brain: `meta_features_vprod.json` + offline OOF scores

#### Benefits

- Deterministic scoring
- Easy replay & debugging
- Isolation from upstream changes

> The Offline Feature Store acts as an in-memory, read-only source of truth for all static and semi-static model features.

---

## 2. Online Feature Store

Implemented in `backend/feature_store.py` as `OnlineFeatureStore`.

### Purpose

Adds **real-time velocity features** based on recent activity (sliding window):

- Sudden bursts of transactions
- Many different recipients in short time
- Many senders to same destination

### Internal Structure

- **user_events**: `cst_dim_id → deque(timestamp, amount, direction)`
- **dir_events**: `direction → deque(timestamp, cst_dim_id, amount)`

### Time Windows

```python
w1 = 60     # last 1 minute
w10 = 600   # last 10 minutes
w60 = 3600  # last 60 minutes
```

### Feature Computation

Called via:
```python
update_and_get_features(cst_id, direction, amount, ts)
```

Returns:

```python
{
  "user_tx_1m": ...,
  "user_tx_10m": ...,
  "user_tx_60m": ...,
  "user_sum_60m": ...,
  "user_new_dirs_60m": ...,
  "dir_tx_60m": ...,
  "dir_unique_senders_60m": ...
}
```

### Integration with `/score_transaction`

1. Offline features fetched from Parquet
2. Online features computed
3. Merged into `full_features`
4. Passed to all brains

---

### Advantages

#### Real-Time Enrichment

- Online store updates with every new transaction
- Reactive to new fraud patterns (e.g. bursts)

#### Stateless API, Stateful Engine

- Client sends only the current transaction
- Online state is kept in memory

#### Simplicity

- Uses `collections.deque`
- Explicit logic
- Transparent and extendable

---

Together, the **Offline** and **Online** Feature Stores form the complete feature layer of ForteShield AI:

- **Offline** captures long-term signals and model outputs
- **Online** brings fresh behavioral context

> The result: stable, fast, and reactive fraud scoring for every transaction.


---

## 3. Scoring Brains

ForteShield AI uses a **layered ensemble of specialized models**, called **brains**. Each brain focuses on a specific type of fraud signal, and their outputs are combined into a final score by the **Meta Brain**.

All brains use a shared `full_features` dictionary, which merges:
- Offline features from `features_offline_v11.parquet`
- Online features from `OnlineFeatureStore`

Feature schemas for each brain are stored in `config/*.json`.

---

### Fast Gate (`score_fast`)

- **Model**: `CatBoostClassifier`
- **Features**: `features_schema_v11.json`
- **Purpose**:
  - Fast risk estimate for every transaction
  - Feeds `risk_fast` live and `risk_fast_oof_v11` offline

**How it works**:
- Handles mixed numeric and categorical features
- Converts missing/invalid values
- Produces a high-speed standalone fraud probability

---

### Graph Brain (`score_graph`)

- **Model**: `CatBoostClassifier`
- **Features**: `graph_brain_features_v11.json`
- **Focus**: Network & reputation analysis

**Captures**:
- Graph-based aggregates
- Fraudulent neighbor connections
- In/out-degree of nodes

Provides `risk_graph`, an independent signal from transaction structure.

---

### Session Brain (`score_sess`)

- **Model**: `CatBoostClassifier`
- **Features**: `session_features_v11.json`
- **Focus**: Session behavior & anomalies

**Analyzes**:
- Device & channel consistency
- Frequency and timing
- Suspicious navigation or robotic behavior

Provides `risk_sess`, useful for detecting account takeover and bot patterns.

---

### AE Brain (`score_ae_meta`)

- **Model**: `Sklearn Pipeline` (Imputer + Scaler + LogisticRegression)
- **Features**: `autoencoder_meta_features_v11.json`
- **Focus**: Anomaly-based fraud detection

**Works with**:
- Autoencoder scores and embeddings
- Reconstruction errors
- Behavior deviations

Returns `risk_ae`, a calibrated score from anomaly-based features.

---

### Sequence Brain (`score_seq_meta`)

- **Model**: `Sklearn Pipeline`
- **Features**: `sequence_meta_features_v11.json`
- **Focus**: Temporal transaction patterns

**Detects**:
- Escalation (micro → large)
- Repetitive cash-out behavior
- Event sequences with fraud risk

Returns:
- `risk_seq`
- `hist_len` (sequence length flag)

---

### Meta Brain (`score_meta_vprod`)

- **Model**: `meta_vprd.pkl` (Sklearn-style)
- **Features**: `meta_features_vprod.json`
- **Focus**: Final aggregation layer

**Input**:
- OOF scores (e.g., `risk_fast_oof_v11`)
- Live Fast Gate output (`risk_fast`)
- Additional meta features

**Output**:
- `risk_meta` → final fraud probability

**Functionality**:
- Learns when to trust which brain
- Adjusts weights based on meta context
- Used in PolicyEngine, case scoring, and explanations

---

### Summary

| Brain          | Focus                  | Type               | Output       |
|----------------|------------------------|--------------------|--------------|
| Fast Gate      | Transactional + Profile| CatBoostClassifier | `risk_fast`  |
| Graph Brain    | Network structure      | CatBoostClassifier | `risk_graph` |
| Session Brain  | Online behavior        | CatBoostClassifier | `risk_sess`  |
| AE Brain       | Anomaly detection      | Sklearn Pipeline   | `risk_ae`    |
| Sequence Brain | Event patterns         | Sklearn Pipeline   | `risk_seq`   |
| Meta Brain     | Aggregation + Meta     | Sklearn Ensemble   | `risk_meta`  |

> The Meta Brain fuses the insights from all brains into a single, calibrated risk score. This drives fraud decisions, case prioritization, and analyst explanations.

---

## 4. Policy Engine

Defined in `backend/policy_engine.py`:

- Reads config from `config/meta_thresholds_vprod.json`
- Supports strategies:
  - Aggressive
  - Balanced
  - Friendly
- Returns:
  - Final decision: `allow` or `alert`
  - Risk type: Normal / Suspicious / Whale
    - Takes into account amount and special multipliers

---

## 5. Case Store

Defined in `backend/case_store.py` using SQLite:

- Stores:
  - `case_id`, `docno`, `cst_dim_id`, `direction`, `amount`, `transdatetime`
  - Brain scores (`risk_fast`, `risk_meta`, etc.)
  - Flags (`ae_high`, `has_seq_history`, etc.)
  - Strategy used and analyst decision
  - Labeling fields for retraining
  - Shadow model scores

- Migrations are handled automatically with `_ensure_column`

---

## 6. Model Registry Backend

Defined in `backend/model_loader.py`, wraps around `config/model_registry.json`:

- For each brain:
  - `champion`, `challenger`, `previous`
  - `shadow_enabled` flag

- Main operations:
  - `load_champion`
  - `promote(brain)`
  - `rollback(brain)`

Model paths can be relative or absolute.

---

## 7. REST API

Key **FastAPI** endpoints:

- `GET /health` – System status and model info
- `POST /score_transaction` – Main scoring:
  - Input: `TransactionInput` (`docno`, `cst_dim_id`, `amount`, etc.)
  - Output: `ScoreResponse` with final risk, decision, strategy
- `GET /cases`, `GET /case/{id}`, `POST /case/{id}/label`
- `GET /strategy`, `GET /strategies`, `POST /strategy/{name}`
- LLM endpoints:
  - `POST /llm/explain_case`
  - `POST /llm/generate_sar`
- Model governance API:
  - `GET /model_registry`
  - `POST /retrain`
  - `POST /reload_models`
  - `POST /promote_model?brain=...`
  - `POST /rollback_model?brain=...`

---

## 8. Shadow Mode

When `shadow_enabled` is ON:
- Challenger scores are computed in parallel
- Written into `risk_fast_shadow`, `risk_meta_shadow`
- The business logic still uses the champion
- The UI compares champion vs shadow for each case

---

## Explanations and LLM Assistant

Defined in `backend/llm_agent.py`:

### SHAP Factors per Brain
- `catboost_reasons_for_row`: CatBoost TreeSHAP
- `shap_top_for_linear`: Linear model contribution (AE, Sequence, Meta)

### Feature Descriptions
- Uses `feature_dictionary.json` to explain:
  - Human label
  - Feature group
  - Risk hints

### Case-Level Reasoning
- `_build_shap_reasons_for_case` (in `main.py`) collects top 30 risk factors

### Prompt Building and Fallback
- `explain_case_text`, `generate_sar_text` build prompts for LLMs
- Uses OpenAI client (default `gpt-4o-mini`) via environment variable
- If no key or error: fallback summary returned for manual SAR writing

> The LLM layer **does not interfere** with decision logic – it works as an **add-on** for analyst context.



# Analyst Console (Streamlit UI)

The frontend is implemented in `ui/app.py` using **Streamlit** with visualizations powered by **Altair**.

---

## Main Tabs

### 1. Dashboard

- Connects to `/cases` and aggregates data over the last `N` days (configured via `DASHBOARD_WINDOW_DAYS`)
- Displays:
  - Total number of cases
  - Alerts and alert rate
  - Time series of cases and alerts
  - Final risk distribution (allow vs alert)
- Supports auto-refresh

### 2. Cases

- Filterable list of cases:
  - By decision
  - By risk range
  - Search by `docno`, `cst_dim_id`, or `label`
- Interactive table with key fields
- Case details view:
  - Amount, Meta Brain score, decision, risk type
  - Champion vs shadow scores (Fast Gate and Meta Brain)
  - JSON and formatted anomaly score
- Labeling interface:
  - `fraud` / `legit`
  - `use_for_retrain` flag
  - Case status
  - Sends labels to `/case/{id}/label`

### 3. LLM Assistant

- Fast search by `docno` or `client_id`
- Case summary view
- Buttons:
  - "Explain decision (LLM)" → `/llm/explain_case`
  - "Generate SAR draft" → `/llm/generate_sar`

### 4. SHAP Insights

- Scans `data/processed/` for SHAP summary files
- Bar charts of top features by mean |SHAP|
- Supports:
  - Fast Gate
  - Graph Brain
  - Session Brain
  - AE
  - Sequence
  - Meta Brain

### 5. Model Governance

- Reads `/model_registry`
- Displays champion/challenger for each brain
- Fetches `/retrain_last_report` for metric comparison
- Visual panel to:
  - Trigger retraining (`/retrain`)
  - Promote or rollback models
  - Reload model registry and backend

> The UI is styled with ForteBank brand colors to align with stakeholder expectations.

---

## Testing and Simulation Scenarios

**scripts/** folder contains utility tools.

### stream_simulator.py

- Reads a cleaned `transactions.csv`
- Normalizes and sorts data
- Sends it to `/score_transaction`
- Modes:
  - `fast`
  - `slow` (with delay)

### red_team.py

- Simulates a fraud scenario:
  - Series of small transfers + large cash-out
- Displays decision and risk for each step

These help test real-time flow and dashboard response.

---

## Retrain Pipeline & Champion/Challenger

### Workflow

1. Analysts label cases and set `use_for_retrain`
2. Click "Retrain challenger"
3. UI calls `/retrain`, backend:
   - Runs `retrain_models.py`
   - Captures logs
   - Loads latest `retrain_report_*.json`
4. UI:
   - Shows metrics, ROC/PR charts, slices
   - Enables promotion if challenger is better

### Model Management Actions

- Promote Fast Gate or Meta Brain
- Rollback to previous version
- All actions update model registry and reload backend

This supports full **model governance** workflows.

---

## Key Strengths of ForteShield AI

### Multi-layer Model Ensemble

- Specialized brains:
  - Transactions
  - Graph
  - History
  - Session
  - Anomalies
- Combined in a final Meta Brain

### Clean Offline/Online Separation

- Offline: Notebooks, features, configs
- Online: Uses only stable artifacts

### Analyst and Compliance Transparency

- Every decision has:
  - Model scores
  - Risk type
  - SHAP insights
- LLM assistant:
  - Explains in human terms
  - Generates SAR drafts (in Russian)

### Operational Readiness

- Shadow mode
- Champion/challenger logic
- Retraining pipeline
- Promotion and rollback
- Built-in labeling

### Full End-to-End Storyline

- From raw data to production-ready platform
- Analysts can:
  - Stream data
  - Label cases
  - Read explanations
  - Manage models
  - Monitor fraud trends

---

## How to Run Locally (Example)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Prepare data
Put processed parquet files into `data/processed/`  
(e.g. `features_offline_v11.parquet`, SHAP summaries)

### Configure and run backend
```bash
export OPENAI_API_KEY="<API key>"
export OPENAI_SAR_MODEL="gpt-4.1-mini"
export OPENAI_EXPLAIN_MODEL="gpt-4.1-mini"

uvicorn backend.main:app --reload
```

### Run UI
```bash
streamlit run ui/app.py
# open http://localhost:8501
```

### (Optional) Run stream simulator
```bash
python scripts/stream_simulator.py
```

---

## Summary

**ForteShield AI** showcases model quality and architectural maturity for anti-fraud at ForteBank:

- Multi-model ensemble including Session Brain
- Meta Brain for risk aggregation
- Real-time velocity and reputation features
- Transparent APIs and governance process
- Intuitive UI for analysts and regulators
- End-to-end tooling for demos and real scenarios
