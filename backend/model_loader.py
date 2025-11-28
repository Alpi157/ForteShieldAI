# backend/model_loader.py

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import json
from datetime import datetime
import logging

from catboost import CatBoostClassifier
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"

logger = logging.getLogger(__name__)


class ModelRegistryBackend:
    """
    Обвязка вокруг config/model_registry.json:

      - загрузка реестра
      - загрузка champion- и (опционально) challenger-моделей
        Fast Gate и Meta Brain
      - promote / rollback champion/challenger

    Ожидаемая структура model_registry.json (упрощённо):

    {
      "fast_gate": {
        "champion":  { "model_path": "models/fast_gate_v11.cbm", "version": "v11" },
        "challenger":{ "model_path": "models/fast_gate_v12.cbm", "version": "v12" },
        "shadow_enabled": true
      },
      "meta_brain": {
        "champion":  { "model_path": "models/meta_vprod.pkl", "version": "vprod" },
        "challenger":{ "model_path": "models/meta_vnext.pkl", "version": "vnext" },
        "shadow_enabled": true
      }
    }
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path: Path = registry_path or (CONFIG_DIR / "model_registry.json")
        self.registry: Dict[str, Any] = {}

        # загруженные champion-модели
        self.fast_gate: Optional[CatBoostClassifier] = None
        self.meta_brain: Any = None

        # загруженные challenger-модели (для shadow-режима / A/B)
        self.fast_gate_challenger: Optional[CatBoostClassifier] = None
        self.meta_brain_challenger: Any = None

        # версионные метки (для логов/UI)
        self.current_fast_version: Optional[str] = None
        self.current_meta_version: Optional[str] = None
        self.challenger_fast_version: Optional[str] = None
        self.challenger_meta_version: Optional[str] = None

        # флаги shadow-режима (читаются из registry)
        self.fast_gate_shadow_enabled: bool = False
        self.meta_brain_shadow_enabled: bool = False

    # ---------- служебное ----------

    def _now_iso(self) -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat()

    def _resolve_path(self, rel_or_abs: str) -> Path:
        """
        Превращаем путь из model_registry в абсолютный относительно корня проекта.
        """
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        return BASE_DIR / rel_or_abs

    # ---------- работа с registry ----------

    def load_registry(self) -> None:
        if not self.registry_path.exists():
            raise FileNotFoundError(f"model_registry.json не найден: {self.registry_path}")
        with open(self.registry_path, "r", encoding="utf-8") as f:
            self.registry = json.load(f)

    def save_registry(self) -> None:
        if not self.registry:
            raise RuntimeError("Нечего сохранять: registry пустой")
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)

    # ---------- загрузка champion (+ challenger) моделей ----------

    def load_champion(self) -> None:
        """
        Загружает champion-версии Fast Gate и Meta Brain согласно model_registry.json.
        Параллельно, если есть, загружает challenger и выставляет флаги shadow_enabled.
        """
        self.load_registry()

        fg_reg = self.registry.get("fast_gate", {})
        mb_reg = self.registry.get("meta_brain", {})

        fg_block = fg_reg.get("champion")
        mb_block = mb_reg.get("champion")

        if fg_block is None or mb_block is None:
            raise ValueError("В model_registry.json нет fast_gate.champion или meta_brain.champion")

        # ---- Fast Gate: champion ----
        fg_model_path = self._resolve_path(fg_block["model_path"])
        if not fg_model_path.exists():
            raise FileNotFoundError(f"Fast Gate модель (champion) не найдена: {fg_model_path}")
        fast_model = CatBoostClassifier()
        fast_model.load_model(str(fg_model_path))

        # ---- Meta Brain: champion ----
        mb_model_path = self._resolve_path(mb_block["model_path"])
        if not mb_model_path.exists():
            raise FileNotFoundError(f"Meta Brain модель (champion) не найдена: {mb_model_path}")
        meta_model = joblib.load(mb_model_path)

        self.fast_gate = fast_model
        self.meta_brain = meta_model
        self.current_fast_version = fg_block.get("version")
        self.current_meta_version = mb_block.get("version")

        # ---- challenger + shadow-флаги ----

        # сбрасываем, чтобы не держать старые ссылки
        self.fast_gate_challenger = None
        self.meta_brain_challenger = None
        self.challenger_fast_version = None
        self.challenger_meta_version = None

        # shadow-enabled флаги
        self.fast_gate_shadow_enabled = bool(fg_reg.get("shadow_enabled", False))
        self.meta_brain_shadow_enabled = bool(mb_reg.get("shadow_enabled", False))

        # Fast Gate challenger
        fg_ch_block = fg_reg.get("challenger")
        if fg_ch_block and fg_ch_block.get("model_path"):
            ch_path = self._resolve_path(fg_ch_block["model_path"])
            if ch_path.exists():
                try:
                    ch_model = CatBoostClassifier()
                    ch_model.load_model(str(ch_path))
                    self.fast_gate_challenger = ch_model
                    self.challenger_fast_version = fg_ch_block.get("version")
                except Exception as e:
                    logger.warning(
                        "Не удалось загрузить Fast Gate challenger (%s): %s",
                        ch_path,
                        e,
                    )
            else:
                logger.warning("Fast Gate challenger модель не найдена: %s", ch_path)

        # Meta Brain challenger
        mb_ch_block = mb_reg.get("challenger")
        if mb_ch_block and mb_ch_block.get("model_path"):
            ch_path = self._resolve_path(mb_ch_block["model_path"])
            if ch_path.exists():
                try:
                    ch_model = joblib.load(ch_path)
                    self.meta_brain_challenger = ch_model
                    self.challenger_meta_version = mb_ch_block.get("version")
                except Exception as e:
                    logger.warning(
                        "Не удалось загрузить Meta Brain challenger (%s): %s",
                        ch_path,
                        e,
                    )
            else:
                logger.warning("Meta Brain challenger модель не найдена: %s", ch_path)

        logger.info(
            "Loaded models from registry: "
            "FastGate champion=%s, challenger=%s, shadow_enabled=%s; "
            "MetaBrain champion=%s, challenger=%s, shadow_enabled=%s",
            self.current_fast_version,
            self.challenger_fast_version,
            self.fast_gate_shadow_enabled,
            self.current_meta_version,
            self.challenger_meta_version,
            self.meta_brain_shadow_enabled,
        )

    # ---------- promote / rollback ----------

    def _swap_blocks(self, brain: str, a_key: str, b_key: str) -> None:
        """
        Переставляет местами под-блоки (champion / challenger / previous) для конкретного brain.
        """
        if not self.registry:
            self.load_registry()

        if brain not in self.registry:
            raise ValueError(f"В registry нет секции '{brain}'")

        block = self.registry[brain]
        a = block.get(a_key)
        b = block.get(b_key)

        if a is None or b is None:
            raise ValueError(f"Нельзя поменять местами {a_key} и {b_key} — один из них пустой")

        block[a_key], block[b_key] = b, a
        self.registry["updated_at"] = self._now_iso()
        self.save_registry()

    def promote(self, brain: str) -> None:
        """
        Делает challenger -> champion, а старый champion уезжает в previous.
        shadow_enabled и прочие поля секции brain не трогаем.
        """
        if not self.registry:
            self.load_registry()

        block = self.registry.get(brain)
        if not block or not block.get("challenger"):
            raise ValueError(f"Для {brain} нет challenger, нечего промоутить")

        old_champion = block.get("champion")
        new_champion = block.get("challenger")

        # сохраняем текущий champion в previous
        block["previous"] = old_champion
        # challenger становится champion
        block["champion"] = new_champion
        # challenger чистим
        block["challenger"] = None

        self.registry["updated_at"] = self._now_iso()
        self.save_registry()

        logger.info(
            "Promote: %s challenger(version=%s) стал champion вместо %s",
            brain,
            (new_champion or {}).get("version") if isinstance(new_champion, dict) else None,
            (old_champion or {}).get("version") if isinstance(old_champion, dict) else None,
        )

    def rollback(self, brain: str) -> None:
        """
        Откат: champion <-> previous.
        """
        if not self.registry:
            self.load_registry()

        block = self.registry.get(brain)
        if not block or not block.get("previous"):
            raise ValueError(f"Для {brain} нет previous, нечего откатывать")

        champion = block.get("champion")
        previous = block.get("previous")
        block["champion"], block["previous"] = previous, champion

        self.registry["updated_at"] = self._now_iso()
        self.save_registry()

        logger.info(
            "Rollback: %s champion(version=%s) <-> previous(version=%s)",
            brain,
            (champion or {}).get("version") if isinstance(champion, dict) else None,
            (previous or {}).get("version") if isinstance(previous, dict) else None,
        )
