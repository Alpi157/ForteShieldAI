from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TransactionInput(BaseModel):
    """
    Входная структура для /score_transaction.

    docno        — идентификатор транзакции (как в оффлайн-фичесторе),
    cst_dim_id   — идентификатор клиента,
    direction    — направление (IN / OUT / ...),
    amount       — сумма транзакции,
    transdatetime — время транзакции.
    """
    docno: int
    cst_dim_id: str
    direction: str
    amount: float
    transdatetime: datetime


class ScoreResponse(BaseModel):
    """
    Базовый ответ скоринга, который используется UI и внешними клинтами.
    Shadow-скоринг challenger-моделей логируется в cases, но не входит сюда.
    """
    case_id: int
    risk_score_final: float
    decision: str
    risk_type: str
    strategy_used: str


class CaseLabelUpdate(BaseModel):
    """
    Патч-разметка для кейса (используется endpoint /case/{case_id}/label).

    user_label:
        - 'fraud'  — подтверждённое мошенничество,
        - 'legit'  — валидный клиент/операция,
        - 'unknown' или None — нет разметки / спорный кейс.

    use_for_retrain:
        - True/False — включать ли кейс в выборку для retrain.

    status:
        - 'new' / 'in_review' / 'closed' / etc — статус обработки кейса.
    """
    user_label: Optional[str] = None
    use_for_retrain: Optional[bool] = None
    status: Optional[str] = None
