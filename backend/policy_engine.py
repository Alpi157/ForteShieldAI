# backend/policy_engine.py

from __future__ import annotations
import json
import os
from typing import Dict, Any


class PolicyEngine:
    """
    Движок стратегий риска.

    Читает пороги из JSON вида:
      - {"strategies": { "Aggressive": {...}, ...}}
      - либо плоский {"Aggressive": {...}, ...}
      - либо {"thresholds": { ... }}

    По risk_final выбирает decision (allow/alert) и риск-тип.
    """

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = os.path.join("config", "meta_thresholds_vprod.json")

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                if "strategies" in data and isinstance(data["strategies"], dict):
                    self.thresholds: Dict[str, Dict[str, Any]] = data["strategies"]
                elif "thresholds" in data and isinstance(data["thresholds"], dict):
                    self.thresholds = data["thresholds"]
                else:
                    self.thresholds = {k: v for k, v in data.items() if isinstance(v, dict)}
            else:
                self.thresholds = {}
        else:
            # запасной вариант
            self.thresholds = {
                "Aggressive": {"threshold": 0.02},
                "Balanced": {"threshold": 0.05},
                "Friendly": {"threshold": 0.1},
            }

        self.current_strategy = "Balanced"
        if self.current_strategy not in self.thresholds and self.thresholds:
            self.current_strategy = next(iter(self.thresholds.keys()))

    def set_strategy(self, name: str) -> None:
        if name not in self.thresholds:
            raise ValueError(f"Unknown strategy: {name}")
        self.current_strategy = name

    def get_strategy(self) -> Dict[str, Any]:
        return {"name": self.current_strategy, "config": self.thresholds[self.current_strategy]}

    def list_strategies(self) -> Dict[str, Any]:
        return {"current": self.current_strategy, "strategies": self.thresholds}

    def apply_policy(self, risk_final: float, features: dict, scores: dict) -> dict:
        """
        Простая логика: порог стратегии -> allow/alert.
        Внутри alert делим на Suspicious/Whale.
        """
        cfg = self.thresholds.get(self.current_strategy, {})
        threshold = float(cfg.get("threshold", 0.05))

        WHALE_MULTIPLIER = 3.0
        whale_cut = min(max(threshold * WHALE_MULTIPLIER, threshold), 0.8)

        if risk_final < threshold:
            decision = "allow"
            risk_type = "Normal"
        else:
            decision = "alert"
            risk_type = "Whale" if risk_final >= whale_cut else "Suspicious"

        return {
            "decision": decision,
            "risk_type": risk_type,
            "threshold_used": threshold,
            "whale_cut": whale_cut,
        }
