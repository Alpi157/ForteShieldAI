# backend/case_store.py

from __future__ import annotations
from typing import Dict, Any, List, Optional
import sqlite3
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CaseStore:
    """
    Простое хранилище кейсов в SQLite.

    Важно:
    - колонка risk_score_final = финальный скор (у нас это risk_meta)
    - decision и risk_type храним явно, чтобы дашборд мог по ним строить метрики
    - user_label / use_for_retrain используются при перетренировке
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    # ======================
    # ИНИЦИАЛИЗАЦИЯ / МИГРАЦИЯ
    # ======================

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        # Базовая схема таблицы cases (для новых БД)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT (datetime('now')),
                docno INTEGER,
                cst_dim_id TEXT,
                direction TEXT,
                amount REAL,
                transdatetime TEXT,

                -- мозги
                risk_fast REAL,
                risk_ae REAL,
                risk_graph REAL,
                risk_seq REAL,
                risk_sess REAL,
                risk_meta REAL,

                -- алиас для фронта (distribution risk_score_final)
                risk_score_final REAL,

                anomaly_score_emb REAL,

                -- флаги high
                ae_high INTEGER,
                graph_high INTEGER,
                seq_high INTEGER,
                sess_high INTEGER,
                has_seq_history INTEGER,

                -- стратегия и решение
                strategy_used TEXT,
                decision TEXT,
                risk_type TEXT,
                status TEXT,

                -- разметка аналитика
                user_label TEXT,
                use_for_retrain INTEGER,

                -- shadow-скоринги (challenger)
                risk_fast_shadow REAL,
                risk_meta_shadow REAL
            )
            """
        )
        self.conn.commit()

        # Для уже существующих БД (старые версии) — добавляем недостающие колонки через ALTER TABLE
        self._ensure_column("status", "TEXT", default="'new'")
        self._ensure_column("user_label", "TEXT", default="NULL")
        self._ensure_column("use_for_retrain", "INTEGER", default="1")
        self._ensure_column("risk_fast_shadow", "REAL", default="NULL")
        self._ensure_column("risk_meta_shadow", "REAL", default="NULL")

    def _ensure_column(self, name: str, col_type: str, default: Optional[str] = None) -> None:
        """
        Проверяет наличие колонки в таблице cases и добавляет её при отсутствии.
        default — SQL-выражение, например '0' или 'NULL' (без ключевого слова DEFAULT).
        """
        cur = self.conn.cursor()
        cur.execute("PRAGMA table_info(cases)")
        existing_cols = {row[1] for row in cur.fetchall()}  # row[1] = name

        if name in existing_cols:
            return

        alter_sql = f"ALTER TABLE cases ADD COLUMN {name} {col_type}"
        if default is not None:
            alter_sql += f" DEFAULT {default}"

        logger.info("Добавляю колонку %s в таблицу cases: %s", name, alter_sql)
        cur.execute(alter_sql)
        self.conn.commit()

    # ======================
    # CRUD
    # ======================

    def create_case(self, data: Dict[str, Any]) -> int:
        """
        Сохраняем кейс. Если risk_score_final не передан, берём из risk_meta.
        user_label / use_for_retrain пока могут быть None — для новых кейсов будет
        расставлять аналитик.
        """
        if data.get("risk_score_final") is None and data.get("risk_meta") is not None:
            data["risk_score_final"] = data["risk_meta"]

        with self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO cases (
                    docno, cst_dim_id, direction, amount, transdatetime,
                    risk_fast, risk_ae, risk_graph, risk_seq, risk_sess,
                    risk_meta, risk_score_final,
                    anomaly_score_emb,
                    ae_high, graph_high, seq_high, sess_high, has_seq_history,
                    strategy_used, decision, risk_type, status,
                    user_label, use_for_retrain,
                    risk_fast_shadow, risk_meta_shadow
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.get("docno"),
                    data.get("cst_dim_id"),
                    data.get("direction"),
                    data.get("amount"),
                    data.get("transdatetime"),
                    data.get("risk_fast"),
                    data.get("risk_ae"),
                    data.get("risk_graph"),
                    data.get("risk_seq"),
                    data.get("risk_sess"),
                    data.get("risk_meta"),
                    data.get("risk_score_final"),
                    data.get("anomaly_score_emb"),
                    data.get("ae_high"),
                    data.get("graph_high"),
                    data.get("seq_high"),
                    data.get("sess_high"),
                    data.get("has_seq_history"),
                    data.get("strategy_used"),
                    data.get("decision"),
                    data.get("risk_type") or "unknown",
                    data.get("status", "new"),
                    data.get("user_label"),
                    int(data.get("use_for_retrain", 1)) if data.get("use_for_retrain") is not None else 1,
                    data.get("risk_fast_shadow"),
                    data.get("risk_meta_shadow"),
                ),
            )
            return cur.lastrowid

    def list_cases(self, limit: int = 100) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT *
            FROM cases
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_case(self, case_id: int) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT * FROM cases WHERE id = ?",
            (case_id,),
        )
        row = cur.fetchone()
        return dict(row) if row is not None else None

    def update_case_label(
        self,
        case_id: int,
        user_label: Optional[str] = None,
        use_for_retrain: Optional[bool] = None,
        status: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Обновляет разметку кейса:
          - user_label: 'fraud' / 'legit' / 'unknown'
          - use_for_retrain: True/False
          - status: 'open' / 'in_review' / 'closed' / и т.п.

        Любой из параметров можно не передавать (None) — тогда поле останется без изменений.
        Возвращает обновлённый кейс или None, если case_id не найден.
        """
        fields = []
        params: List[Any] = []

        if user_label is not None:
            fields.append("user_label = ?")
            params.append(user_label)

        if use_for_retrain is not None:
            fields.append("use_for_retrain = ?")
            params.append(int(bool(use_for_retrain)))

        if status is not None:
            fields.append("status = ?")
            params.append(status)

        if not fields:
            # Нечего обновлять — просто вернём текущую запись
            return self.get_case(case_id)

        params.append(case_id)

        with self.conn:
            self.conn.execute(
                f"UPDATE cases SET {', '.join(fields)} WHERE id = ?",
                params,
            )

        logger.info(
            "Обновлена разметка кейса id=%s (user_label=%s, use_for_retrain=%s, status=%s)",
            case_id,
            user_label,
            use_for_retrain,
            status,
        )

        return self.get_case(case_id)
