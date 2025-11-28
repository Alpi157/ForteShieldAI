# ForteShield AI — многоуровневая антифрод-платформа для ForteBank

ForteShield AI — это сквозное end-to-end решение **от сырых данных до рабочего места аналитика**:  
от оффлайн-обучения ансамбля моделей до онлайн-скоринга транзакций в real-time, дашборда для антифрод-команды и контура model governance с retrain, champion/challenger и LLM-ассистентом.

---

## Содержание

- [Бизнес-контекст и сценарий использования](#бизнес-контекст-и-сценарий-использования)
- [Архитектура решения](#архитектура-решения)  
  - [Структура репозитория](#структура-репозитория)
- [Онлайн-скоринг и backend (FastAPI)](#онлайн-скоринг-и-backend-fastapi)  
  - [Offline Feature Store](#1-offline-feature-store)  
  - [OnlineFeatureStore](#2-onlinefeaturestore)  
  - [Мозги скоринга (Brains)](#3-мозги-скоринга-brains)  
  - [PolicyEngine](#4-policyengine)  
  - [CaseStore](#5-casestore)  
  - [ModelRegistryBackend и model governance](#6-modelregistrybackend-и-model-governance)  
  - [REST API](#7-rest-api)  
  - [Shadow-режим](#8-shadow-режим)
- [Эксплейны и LLM-ассистент](#эксплейны-и-llm-ассистент)
- [Консоль аналитика (Streamlit UI)](#консоль-аналитика-streamlit-ui)
- [Сценарии тестирования и симуляции](#сценарии-тестирования-и-симуляции)
- [Retrain-pipeline и champion/challenger](#retrain-pipeline-и-championchallenger)
- [Ключевые достоинства ForteShield AI](#ключевые-достоинства-forteshield-ai)
- [Как запустить локально (пример)](#как-запустить-локально-пример)
- [Итог](#итог)

---

## Бизнес-контекст и сценарий использования

**Исходный кейс** — мошеннические транзакции в ForteBank:

- финансовые операции клиентов (классический транзакционный датасет);
- дополнительный датасет с поведенческими/сессионными данными (логины, действия в онлайне, поведение в приложении).

ForteShield AI отвечает на несколько практических задач:

1. **Быстрое решение по каждой операции**  
   Для каждой входящей транзакции нужно за миллисекунды оценить риск и принять решение:
   - пропустить;
   - поднять алерт;
   - выделить особо крупный риск («whale»).

2. **Глубокая аналитика по кейсам**  
   Аналитик должен:
   - видеть приоритетную очередь кейсов;
   - понимать, *почему* модель приняла решение;
   - иметь удобный интерфейс для разметки (fraud / legit) и последующего retrain.

3. **Управление моделями**  
   Команде риска нужна не одна «зашитая» модель, а:
   - новые версии моделей на свежей разметке;
   - понятный **champion/challenger**;
   - **shadow-режим** и безопасный откат.

4. **Поддержка комплаенса и SAR**  
   Для сложных кейсов нужны:
   - текстовые объяснения на человеческом языке;
   - черновики SAR-отчётов, готовые к адаптации под регулятора.

**ForteShield AI** покрывает весь этот цикл в рамках **одной репозитории**.

---

## Архитектура решения

На верхнем уровне проект делится на несколько слоёв:

- `notebooks/` — оффлайн-pipeline построения фич и обучения моделей (v11 / vprod);
- `backend/` — FastAPI-сервис скоринга и model governance;
- `ui/` — Streamlit-консоль аналитика (дашборд, LLM-ассистент, model governance);
- `models/` — артефакты моделей (CatBoost и sklearn-пайплайны);
- `config/` — схемы фич, пороги, `model_registry.json`, словарь признаков;
- `data/processed/` — оффлайн-фичестор и SHAP-сводки;
- `scripts/` — утилиты для симуляции потока и red-team сценариев;
- `retrain_models.py` — скрипт перетренировки challenger-моделей (вызывается из backend и UI).

В продакшн-аналогии:

- `notebooks/` и `retrain_models.py` — зона ответственности DS-команды;
- `backend/` + `ui/` — зона ML/инженеров и антифрод-аналитиков.

### Структура репозитория

````text
.
├── notebooks/           # Оффлайн-EDA, фичи, обучение моделей, SHAP
├── backend/             # FastAPI-сервис скоринга и model governance
│   ├── main.py          # Точка входа, REST API
│   ├── feature_store.py # OnlineFeatureStore
│   ├── policy_engine.py # PolicyEngine со стратегиями риска
│   ├── case_store.py    # SQLite-хранилище кейсов
│   ├── model_loader.py  # ModelRegistryBackend
│   ├── llm_agent.py     # LLM-эксплейны и SAR-черновики
│   └── ...
├── ui/
│   └── app.py           # Streamlit-консоль аналитика
├── models/              # CatBoost, sklearn-пайплайны, Meta Brain
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
└── retrain_models.py    # Перетренировка challenger-моделей
````

---

## Онлайн-скоринг и backend (FastAPI)

Корневой сервис расположен в `backend/main.py` и поднимается как FastAPI-приложение.

### 1. Offline Feature Store

* `features_offline_v11.parquet` загружается в память один раз при старте.
* Индексация по `docno` обеспечивает:
  * стабильность фичей;
  * повторяемость для демо и оффлайн-анализа.

### 2. OnlineFeatureStore

Файл `backend/feature_store.py` реализует лёгкий **in-memory** стор:

* для каждого клиента и получателя хранит последние события;
* считает velocity-фичи:
  * `user_tx_1m / 10m / 60m`
  * `user_sum_60m`
  * `user_new_dirs_60m`
  * `dir_tx_60m`
  * `dir_unique_senders_60m`

Фичи автоматически обновляются при каждом запросе к `/score_transaction`.

### 3. Мозги скоринга (Brains)

В `main.py` явно разделены функции скоринга:

* **Fast Gate** — `score_fast`  
  CatBoostClassifier с числовыми и категориальными признаками.  
  Категориальные индексы берутся из конфигурации, пропуски аккуратно обрабатываются.

* **Graph Brain** — `score_graph`  
  CatBoost по графовым и репутационным фичам.

* **Session Brain** — `score_sess`  
  Модель по сессионным и поведенческим признакам, умеет обрабатывать смешанные типы.

* **AE Brain** — `score_ae_meta`  
  sklearn-пайплайн (imputer, scaler, LogisticRegression) по аномалиям.

* **Sequence Brain** — `score_seq_meta`  
  Пайплайн по последовательностям транзакций, дополнительно возвращающий длину истории.

* **Meta Brain** — `score_meta_vprod`  
  Финальная модель `risk_meta_vprod.pkl`, которая использует:
  * оффлайн OOF-скоры мозгов:
    * `risk_fast_oof_v11`
    * `risk_ae_oof_v11`
    * `graph_brain_oof_v11`
    * `risk_seq_oof_v11`
    * `risk_sess_oof_v11`
  * live-скор Fast Gate;
  * набор мета-фич из `meta_features_vprod.json`.

Результат Meta Brain — финальная вероятность фрода, которая далее идёт в **PolicyEngine**.

### 4. PolicyEngine

Файл `backend/policy_engine.py`:

* читает конфигурацию порогов из `config/meta_thresholds_vprod.json`;
* поддерживает несколько стратегий:
  * `Aggressive`
  * `Balanced`
  * `Friendly`
  * и др.
* возвращает:
  * итоговое решение (`allow` / `alert`);
  * тип риска: `Normal / Suspicious / Whale` с учётом:
    * размера суммы;
    * специального множителя для крупных операций.

### 5. CaseStore

Файл `backend/case_store.py` — лёгкое, но продуманное хранилище кейсов на **SQLite**.

Таблица `cases` хранит:

* базовые поля:
  * `case_id`, `docno`, `cst_dim_id`, `direction`, `amount`, `transdatetime`;
* скоры мозгов:
  * `risk_fast`, `risk_ae`, `risk_graph`, `risk_seq`, `risk_sess`, `risk_meta`;
  * `risk_score_final`, `anomaly_score_emb`;
* флаги:
  * `ae_high`, `graph_high`, `seq_high`, `sess_high`, `has_seq_history`;
* стратегия и решение:
  * `strategy_used`, `decision`, `risk_type`, `status`;
* разметка:
  * `user_label`, `use_for_retrain`;
* поля shadow-режима:
  * `risk_fast_shadow`, `risk_meta_shadow`.

Миграции реализованы через `_ensure_column`: при обновлении репо таблица догоняется через `ALTER TABLE`.

### 6. ModelRegistryBackend и model governance

Файл `backend/model_loader.py` — обвязка над `config/model_registry.json`.

Для каждого brain (например, `fast_gate`, `meta_brain`) описано:

* `champion` — текущая боевая модель;
* `challenger` — новая версия на свежих данных;
* `previous` — предыдущий champion для откатов;
* `shadow_enabled` — флаг shadow-режима.

Основные операции:

* `load_champion` — загрузить champion и (опционально) challenger;
* `promote(brain)` — `challenger → champion`, старый champion уезжает в `previous`;
* `rollback(brain)` — обмен `champion ↔ previous`.

Пути к моделям могут быть относительными (от корня проекта) или абсолютными.



### 7. REST API

Ключевые эндпоинты FastAPI:

* `GET /health`  
  Короткий healthcheck + версии champion-моделей и статус shadow-режима.

* `POST /score_transaction`  
  Главный endpoint скоринга.  
  Принимает `TransactionInput`:
  * `docno`, `cst_dim_id`, `direction`, `amount`, `transdatetime`.

  Пайплайн:
  1. Поднимает оффлайн-фичи по `docno`;
  2. Добавляет online-velocity;
  3. Считает все мозги;
  4. Собирает финальный риск через Meta Brain;
  5. Прогоняет через PolicyEngine;
  6. Сохраняет кейс в CaseStore.

  Возвращает `ScoreResponse`:
  * `case_id`, `risk_score_final`, `decision`, `risk_type`, `strategy_used`.

* `GET /cases`, `GET /case/{id}`, `POST /case/{id}/label`  
  Очередь кейсов, карточка кейса, обновление разметки `user_label` / `use_for_retrain` / `status`.

* `GET /strategy`, `GET /strategies`, `POST /strategy/{name}`  
  Управление стратегиями PolicyEngine из UI.

* **LLM-эндпоинты**:
  * `POST /llm/explain_case` — текстовое объяснение решения;
  * `POST /llm/generate_sar` — черновик SAR-отчёта.

* **Model governance API**:
  * `GET /model_registry` — текущее содержимое `model_registry.json`;
  * `GET /retrain_last_report` — последний `retrain_report_*.json` из `reports/`;
  * `POST /retrain` — запуск `retrain_models.py`, возврат stdout/stderr и отчёта;
  * `POST /reload_models` — перечитать champion/challenger из `model_registry.json`;
  * `POST /promote_model?brain=fast_gate|meta_brain`;
  * `POST /rollback_model?brain=...`.

---

### 8. Shadow-режим

При включённом `shadow_enabled` для Fast Gate и Meta Brain:

* backend параллельно считает **challenger-скор**;
* пишет его в кейс (`risk_fast_shadow`, `risk_meta_shadow`);
* бизнес-решение при этом остаётся на champion.

UI визуализирует champion vs shadow для каждого кейса и помогает принять решение о `promote`.

---

## Эксплейны и LLM-ассистент

Файл `backend/llm_agent.py` реализует слой объяснений вокруг моделей.

1. **SHAP-факторы по каждому мозгу**

   * `catboost_reasons_for_row` использует CatBoost TreeSHAP;
   * `shap_top_for_linear` оценивает вклад признаков в логистических пайплайнах (AE, Sequence, Meta);
   * `describe_feature` обогащает признак данными из `feature_dictionary.json`:
     * человеко-читаемый **label**;
     * группа признака;
     * короткое описание;
     * подсказки по высокому и низкому риску.

2. **Сбор общих причин по кейсу**

   * `_build_shap_reasons_for_case` в `main.py`:
     * поднимает оффлайн-фичи по `docno`;
     * прогоняет их через Fast, Graph, Session, AE, Sequence и Meta Brain;
     * собирает до **30** наиболее значимых факторов риска с указанием:
       * `brain`;
       * группы признака;
       * текстовых подсказок.

3. **Формирование промптов и fallback**

   * `explain_case_text` и `generate_sar_text` собирают:
     * сводку кейса (сумма, направление, скоры всех мозгов, итоговый риск);
     * описание стратегии и порогов;
     * список факторов риска, сгруппированных по мозгам.
   * Используется современный OpenAI-клиент, модель по умолчанию — `gpt-4o-mini` (конфигурируется через переменную окружения).
   * При отсутствии API-ключа или ошибке:
     * возвращается информативный **fallback** с сырой сводкой и факторами;
     * аналитик может вручную написать объяснение или SAR, опираясь на эти данные.

LLM-слой работает как надстройка над числовыми сигналами **и не вмешивается в само решение по кейсу**.

---

## Консоль аналитика (Streamlit UI)

Фронт-часть реализована в `ui/app.py` на базе **Streamlit** с визуализацией через **Altair**.

Основные вкладки:

### 1. Dashboard

* Подключается к `/cases`, агрегирует данные за последние `N` дней  
  (окно настраивается через `DASHBOARD_WINDOW_DAYS`).
* Показывает:
  * общее число кейсов в окне;
  * количество алертов и `alert_rate`;
  * тайм-серии: число кейсов и алертов по дням;
  * распределение итогового риска с раскладкой по `allow/alert`.
* Автообновление каждые несколько секунд.

### 2. Cases

* Список кейсов с фильтрами по:
  * решению;
  * диапазону риска;
  * поиску по `docno` / `cst_dim_id` / `label`.
* Интерактивная таблица с ключевыми полями.
* Карточка выбранного кейса:
  * сумма, итоговый Meta Brain, решение, тип риска;
  * champion vs shadow-скоры Fast Gate и Meta Brain;
  * полный JSON-вид кейса;
  * аккуратное отображение `anomaly_score_emb` (даже если сейчас не используется).

Интерфейс разметки:

* выбор `label` (`fraud` / `legit`);
* флаг `use_for_retrain`;
* статус кейса (`new`, `in_review`, `closed`);
* отправка изменений через `/case/{id}/label`.

### 3. LLM Assistant

* Быстрый поиск кейсов по `docno` или `client_id`;
* Выбор кейса с кратким summary;
* Две кнопки:
  * «Объяснить решение (LLM)» → `/llm/explain_case`;
  * «Сгенерировать черновик SAR» → `/llm/generate_sar`;
* Результат выводится прямо в UI, при ошибке — понятное сообщение.

### 4. SHAP insights

* Автоматически ищет parquet-файлы SHAP-сводок в `data/processed/` для каждого мозга;
* Строит бар-чарты по `mean |SHAP|`, показывая топ-фичи:
  * Fast Gate;
  * Graph Brain;
  * Session Brain;
  * AE;
  * Sequence;
  * Meta Brain.
* Если файл не найден — показывает ожидаемый путь, чтобы DS-команда могла воспроизвести.

### 5. Model Governance

* Читает `/model_registry` и показывает:
  * champion и challenger версии Fast Gate и Meta Brain;
  * сырой JSON registry (по клику);
* Читает `/retrain_last_report`:
  * строит табличное сравнение метрик champion vs challenger;
  * визуализирует результаты графиками;
* Панель операций:
  * запуск retrain через `/retrain`;
  * `promote` Fast Gate и Meta Brain challenger → champion;
  * `rollback` champion ↔ previous для каждого мозга;
  * после каждой операции UI:
    * перезагружает model_registry;
    * вызывает `/reload_models` в backend.

Всё это оформлено в визуальном стиле, близком к бренд-цветам ForteBank — консоль выглядит как готовый продукт для демонстрации бизнесу.



## Сценарии тестирования и симуляции

Каталог `scripts/` содержит вспомогательные утилиты:

* `stream_simulator.py`
  * читает очищенный `transactions.csv` (из работы участника A);
  * нормализует типы (`datetime`, `docno`, `amount`, направления);
  * сортирует по времени;
  * отправляет транзакции в backend через `/score_transaction`;
  * поддерживает режимы:
    * `fast`;
    * `slow` (с имитацией задержек).

* `red_team.py`
  * сценарий «micro-fraud + cash-out»:
    * серия мелких транзакций;
    * один крупный вывод;
  * прогоняет цепочку через backend;
  * печатает для каждой операции:
    * решение;
    * риск.

Эти скрипты позволяют воспроизводить потоковые сценарии, наблюдать метрики на Dashboard и анализировать, какие кейсы попадают в очередь.

---

## Retrain-pipeline и champion/challenger

Цепочка перетренировки:

1. Аналитики размечают кейсы в UI и отмечают `use_for_retrain`.

2. По кнопке «Перетренировать challenger» во вкладке Model Governance вызывается `/retrain`.

3. Backend:
   * запускает `retrain_models.py` синхронно;
   * собирает `stdout`/`stderr`;
   * после завершения читает последний `retrain_report_*.json` из `reports/`.

4. В отчёте:
   * сравнение champion vs challenger для Fast Gate и Meta Brain:
     * метрики до/после;
     * ROC/PR;
     * срезы по таргету и т.д.

5. UI:
   * показывает таблицу и графики сравнений;
   * позволяет раскрыть сырой JSON отчёта.

6. Если новые модели лучше:
   * аналитик нажимает:
     * «Promote Fast Gate challenger → champion» или
     * «Promote Meta Brain challenger → champion»;
   * backend:
     * обновляет `model_registry.json` через `ModelRegistryBackend.promote`;
     * перезагружает модели.

7. При необходимости всегда доступен `Rollback` для каждого мозга.

Так в демо реализована культура **model governance**, к которой привыкли банки.

---

## Ключевые достоинства ForteShield AI

1. **Многоуровневый ансамбль моделей**  
   Несколько специализирующихся «мозгов»:
   * транзакции;
   * граф связей;
   * история клиента;
   * поведение в сессии;
   * аномалии  
     → плюс финальный Meta Brain дают более надёжный сигнал, чем одна монолитная модель.

2. **Чистая граница между оффлайном и онлайном**
   * оффлайн-фичестор, OOF-скоры, SHAP-сводки, конфиги фич — в ноутбуках;
   * онлайн-сервис опирается на стабильные артефакты;
   * в коде backend нет «зашитых» фичей — всё берётся из артефактов и конфигов.

3. **Прозрачность для аналитиков и комплаенса**
   * каждое решение сопровождается:
     * скорингами всех мозгов;
     * типом риска;
     * SHAP-факторами;
   * LLM-ассистент:
     * объясняет решения человеческим языком;
     * формирует черновики SAR-отчётов на русском, опираясь на реальные числа под капотом.

4. **Готовность к эксплуатации и развитию**
   * champion/challenger;
   * shadow-режим;
   * retrain-pipeline;
   * `model_registry`;
   * `promote` и `rollback`;
   * разметка кейсов и отбор для retrain — всё выстроено в единый процесс.

5. **Полный end-to-end сторилайн**
   Репозиторий показывает путь:
   * от сырого датасета и исследовательских ноутбуков;
   * до работающего backend-сервиса;
   * и понятной консоли для антифрод-команды.

   Любой член жюри или заказчик может:
   * поднять backend;
   * открыть UI;
   * прогнать поток транзакций;
   * увидеть дашборд;
   * открыть кейсы;
   * посмотреть объяснения;
   * промоутить новую версию модели.

---

## Как запустить локально (пример)

> Команды могут отличаться в зависимости от вашего окружения. Ниже — примерный сценарий.

1. **Установить зависимости**

```bash
pip install -r requirements.txt
```

2. **Подготовить данные**
* положить подготовленные parquet-файлы в `data/processed/`  
  (например, `features_offline_v11.parquet` и SHAP-сводки).

3. **Настройка LLM и запуск backend (FastAPI)**

```bash
$env:OPENAI_API_KEY="<Ключ API>"
$env:OPENAI_SAR_MODEL="gpt-4.1-mini"
$env:OPENAI_EXPLAIN_MODEL="gpt-4.1-mini"

uvicorn backend.main:app --reload
```

4. **Запустить UI (Streamlit)**

```bash
streamlit run ui/app.py
# откроется в браузере: http://localhost:8501
```

5. **(Опционально) Запустить симулятор потока**

```bash
python scripts/stream_simulator.py
```

## Итог

ForteShield AI демонстрирует не только качество моделей, но и зрелость всей архитектуры антифрод-решения для ForteBank.

В рамках одного проекта реализованы:

* ансамбль специализированных моделей, включая Session Brain на поведенческих данных;
* финальный Meta Brain, собирающий их в единый риск-скор;
* online-фичи по velocity и репутации контрагентов;
* стабильный feature store и прозрачный оффлайн-pipeline;
* backend-сервис на FastAPI с понятным API;
* консоль аналитика с очередью кейсов, фильтрами, дашбордом и LLM-ассистентом;
* культура model governance с retrain, champion/challenger, shadow-режимом, promote и rollback.
