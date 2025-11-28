from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Dict, Deque, Tuple, Any


class OnlineFeatureStore:
    """
    Простой in-memory стор для online-фичей (velocity и т.п.).

    Для каждого клиента и получателя храним последние события
    и на их основе считаем:
      - user_tx_1m / 10m / 60m
      - user_sum_60m
      - user_new_dirs_60m
      - dir_tx_60m
      - dir_unique_senders_60m
    """

    def __init__(self, w1: int = 60, w10: int = 600, w60: int = 3600) -> None:
        # cst_dim_id -> deque[(timestamp, amount, direction)]
        self.user_events: Dict[str, Deque[Tuple[float, float, str]]] = {}
        # direction -> deque[(timestamp, cst_dim_id, amount)]
        self.dir_events: Dict[str, Deque[Tuple[float, str, float]]] = {}
        self.w1, self.w10, self.w60 = w1, w10, w60

    @staticmethod
    def _cleanup(dq: Deque[Tuple[float, Any, Any]], now_ts: float, window: int) -> None:
        while dq and now_ts - dq[0][0] > window:
            dq.popleft()

    def update_and_get_features(
        self,
        cst_id: str,
        direction: str,
        amount: float,
        ts: datetime,
    ) -> Dict[str, float]:
        """
        Обновляем историю по клиенту/получателю и возвращаем свежие online-фичи.
        """
        t = ts.timestamp()

        # история по клиенту
        dq_u = self.user_events.setdefault(cst_id, deque())
        dq_u.append((t, amount, direction))

        # история по получателю
        dq_d = self.dir_events.setdefault(direction, deque())
        dq_d.append((t, cst_id, amount))

        # чистим окна
        for window in (self.w1, self.w10, self.w60):
            self._cleanup(dq_u, t, window)
            self._cleanup(dq_d, t, window)

        feats: Dict[str, float] = {}

        # -------- фичи по клиенту --------
        times_user = [tt for tt, _, _ in dq_u]

        feats["user_tx_1m"] = sum(1 for tt in times_user if t - tt <= self.w1)
        feats["user_tx_10m"] = sum(1 for tt in times_user if t - tt <= self.w10)
        feats["user_tx_60m"] = sum(1 for tt in times_user if t - tt <= self.w60)

        feats["user_sum_60m"] = sum(a for tt, a, _ in dq_u if t - tt <= self.w60)

        # новые/уникальные получатели за 60 минут
        recent_dirs_60m = {d for tt, _, d in dq_u if t - tt <= self.w60}
        feats["user_new_dirs_60m"] = float(len(recent_dirs_60m))

        # -------- фичи по получателю --------
        times_dir = [tt for tt, _, _ in dq_d]
        senders_dir = [cid for tt, cid, _ in dq_d if t - tt <= self.w60]

        feats["dir_tx_60m"] = sum(1 for tt in times_dir if t - tt <= self.w60)
        feats["dir_unique_senders_60m"] = float(len(set(senders_dir)))

        return feats
