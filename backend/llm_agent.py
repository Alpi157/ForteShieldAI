# backend/llm_agent.py

from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
import math

try:
    # новая библиотека OpenAI (pip install openai)
    from openai import OpenAI
except ImportError:  # если библиотеку не поставили
    OpenAI = None  # type: ignore

_client: Optional["OpenAI"] = None  # type: ignore


def _get_client() -> Optional["OpenAI"]:  # type: ignore
    """
    Ленивая инициализация OpenAI-клиента.
    Если нет OPENAI_API_KEY или библиотеки, вернёт None (будет фолбэк).
    """
    global _client
    if OpenAI is None:
        return None

    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    _client = OpenAI(api_key=api_key)
    return _client


# =========================
# СВОДКА ПО КЕЙСУ / ПОЛИТИКЕ
# =========================

def _build_case_summary(case: Dict[str, Any]) -> str:
    """
    Краткое текстовое резюме кейса для промпта.
    """
    parts = [
        f"docno: {case.get('docno')}",
        f"case_id: {case.get('id')}",
        f"client_id (cst_dim_id): {case.get('cst_dim_id')}",
        f"direction: {case.get('direction')}",
        f"amount (KZT): {case.get('amount')}",
        f"datetime: {case.get('transdatetime')}",
        "",
        f"risk_fast: {case.get('risk_fast')}",
        f"risk_ae: {case.get('risk_ae')}",
        f"risk_graph: {case.get('risk_graph')}",
        f"risk_seq: {case.get('risk_seq')}",
        f"risk_sess: {case.get('risk_sess')}",
        f"risk_meta (final): {case.get('risk_meta')}",
        "",
        f"risk_score_final (legacy): {case.get('risk_score_final')}",
        f"decision: {case.get('decision')}",
        f"risk_type: {case.get('risk_type')}",
        f"strategy_used: {case.get('strategy_used')}",
    ]
    return "\n".join(str(p) for p in parts)


def _build_policy_summary(extra_payload: Optional[Dict[str, Any]]) -> str:
    """
    Кратко описываем политику принятия решения, если она передана.
    """
    if not extra_payload:
        return "Информация о политике (порогах) недоступна."

    policy = extra_payload.get("policy") or {}
    strategy = policy.get("strategy")
    decision = policy.get("decision")
    risk_type = policy.get("risk_type")
    thresholds = policy.get("thresholds") or {}

    lines: List[str] = []
    lines.append(f"Текущая стратегия: {strategy}")
    lines.append(f"Решение по кейсу: {decision}, тип риска: {risk_type}")
    if thresholds:
        lines.append("Конфигурация порогов/стратегии (сырые значения):")
        # Без глубокого парсинга — просто даём JSON-подобный вид
        lines.append(str(thresholds))
    else:
        lines.append("Пороговые значения стратегии не переданы.")

    return "\n".join(lines)


# =========================
# ФОРМАТИРОВАНИЕ REASONS / SHAP
# =========================

def _format_reasons_by_brain(
    reasons: List[Dict[str, Any]],
    extra_payload: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Новый формат: reasons — это список словарей вида
    {
        "feature": ...,
        "raw_value": ...,
        "shap": ...,
        "direction": "up"/"down",
        "group": ...,
        "label": ...,
        "short_desc": ...,
        "risk_hint": ...,
        "brain": "meta_brain"/"fast_gate"/...
    }

    Группируем по мозгам и печатаем человека-понятный текст.
    """
    lines: List[str] = []

    # Сначала кратко покажем глобальные скоры, если есть
    if extra_payload and "global_scores" in extra_payload:
        gs = extra_payload["global_scores"]
        lines.append("Скоринговые модели (вероятность фрода от 0 до 1):")
        for brain_code, data in gs.items():
            score = data.get("score")
            if score is None:
                continue
            try:
                score_f = float(score)
                score_str = f"{score_f:.4f}"
            except Exception:
                score_str = str(score)
            lines.append(f"  - {brain_code}: {score_str}")
        lines.append("")

    if not reasons:
        lines.append("Нет детализированных SHAP-причин для этого кейса.")
        return "\n".join(lines)

    # Группируем по brain
    by_brain: Dict[str, List[Dict[str, Any]]] = {}
    for r in reasons:
        brain = r.get("brain", "other")
        by_brain.setdefault(brain, []).append(r)

    brain_order = [
        "meta_brain",
        "fast_gate",
        "graph_brain",
        "session_brain",
        "ae_brain",
        "sequence_brain",
        "other",
    ]
    brain_titles = {
        "meta_brain": "Meta Brain (итоговая модель)",
        "fast_gate": "Fast Gate (скоринг транзакции)",
        "graph_brain": "Graph Brain (связи/репутация получателей)",
        "session_brain": "Session Brain (поведение в сессии)",
        "ae_brain": "AE Brain (аномалии поведения клиента)",
        "sequence_brain": "Sequence Brain (история транзакций)",
        "other": "Прочие факторы",
    }

    for brain in brain_order:
        if brain not in by_brain:
            continue
        items = by_brain[brain]
        # сортируем по |shap|
        items = sorted(items, key=lambda r: abs(r.get("shap", 0.0)), reverse=True)
        if brain == "meta_brain":
            items = items[:7]
        else:
            items = items[:3]

        lines.append(brain_titles.get(brain, brain_titles["other"]) + ":")
        if not items:
            lines.append("  (нет значимых факторов)")
            lines.append("")
            continue

        for r in items:
            label = r.get("label") or r.get("feature")
            feature = r.get("feature")
            value = r.get("raw_value")
            direction = r.get("direction")
            group = r.get("group") or "other"
            risk_hint = r.get("risk_hint") or ""
            short_desc = r.get("short_desc") or ""

            try:
                shap_val = float(r.get("shap", 0.0))
                shap_mag = abs(shap_val)
                shap_str = f"{shap_val:+.3f} (|SHAP|≈{shap_mag:.3f})"
            except Exception:
                shap_str = str(r.get("shap"))

            if direction == "up":
                dir_text = "повышает риск"
            elif direction == "down":
                dir_text = "снижает риск"
            else:
                dir_text = "влияет на риск"

            value_str = repr(value)

            desc_parts: List[str] = []
            if short_desc:
                desc_parts.append(short_desc)
            if risk_hint:
                desc_parts.append(risk_hint)
            desc = " ".join(desc_parts).strip()

            line = f"  - [{group}] {label} (feature='{feature}', значение={value_str}) {dir_text}; вклад {shap_str}."
            if desc:
                line += f" {desc}"
            lines.append(line)
        lines.append("")

    return "\n".join(lines).rstrip()


def _format_shap_for_prompt(
    top_features: List[Dict[str, Any]] | None,
    top_groups: List[Dict[str, Any]] | None,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Превращаем SHAP-таблички / reasons в текст для промпта.
    - Новый режим: если в top_features есть поле 'brain', считаем это форматированными reasons.
    - Старый режим: просто печатаем ключ=значение.
    """
    top_features = top_features or []
    top_groups = top_groups or []

    # Новый формат reasons (с brain и пр.)
    if top_features and isinstance(top_features[0], dict) and "brain" in top_features[0]:
        return _format_reasons_by_brain(top_features, extra_payload)

    # Старый / fallback формат
    lines: List[str] = []

    if top_features:
        lines.append("Топ-фичи (SHAP):")
        for row in top_features:
            parts = []
            for k, v in row.items():
                if k.lower() in {"docno", "id"}:
                    continue
                parts.append(f"{k}={v}")
            lines.append("  - " + ", ".join(parts))

    if top_groups:
        lines.append("")
        lines.append("Топ-группы признаков (SHAP-группы):")
        for row in top_groups:
            parts = []
            for k, v in row.items():
                if k.lower() in {"docno", "id"}:
                    continue
                parts.append(f"{k}={v}")
            lines.append("  - " + ", ".join(parts))

    return "\n".join(lines) if lines else "Нет SHAP-данных для этого кейса."


# =========================
# LLM: ОБЪЯСНЕНИЕ КЕЙСА
# =========================

def explain_case_text(
    case: Dict[str, Any],
    shap_top_features: List[Dict[str, Any]] | None = None,
    shap_top_groups: List[Dict[str, Any]] | None = None,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> str:
    """
    LLM-объяснение решения по кейсу.
    Если нет доступа к API — возвращаем понятный фолбэк.
    """
    summary = _build_case_summary(case)
    policy_text = _build_policy_summary(extra_payload)
    shap_text = _format_shap_for_prompt(shap_top_features or [], shap_top_groups or [], extra_payload)

    client = _get_client()
    if client is None:
        # Фолбэк без LLM
        return (
            "LLM (ChatGPT API) не сконфигурирован.\n\n"
            "Ниже отладочная сводка по кейсу, политике и факторам риска:\n\n"
            f"=== КЕЙС ===\n{summary}\n\n"
            f"=== ПОЛИТИКА И ПОРОГИ ===\n{policy_text}\n\n"
            f"=== ФАКТОРЫ (SHAP / REASONS) ===\n{shap_text}"
        )

    model = os.getenv("OPENAI_EXPLAIN_MODEL", "gpt-4o-mini")

    user_prompt = f"""
Ниже описан кейс по подозрительной транзакции и выводы скоринговых моделей антифрода.

=== КЕЙС (СЫРЫЕ ДАННЫЕ) ===
{summary}

=== ПОЛИТИКА И ПОРОГИ ===
{policy_text}

=== ФАКТОРЫ РИСКА (SHAP / REASONS) ===
{shap_text}

Твоя задача как аналитика:
1. Кратко (3–5 предложений) опиши ситуацию: кто клиент, что за транзакция, какой итоговый риск (risk_meta) и какое решение принято (decision).
2. Сформируй нумерованный список из 3–7 ключевых факторов риска простым человеческим языком:
   - Для каждого фактора объясни, что он означает (на уровне "часто переводит на подозрительного получателя", "аномальное поведение в мобильном приложении", "высокая доля фродовых операций по контрагенту" и т.п.).
   - Укажи, почему этот фактор повышает или снижает риск именно в этом кейсе (опирайся на значение признака и комментарии в описании).
   - Не используй термин "SHAP" в тексте для пользователя.
3. Объясни роль разных моделей:
   - Как соотносятся оценки Fast Gate, Graph Brain, Session Brain, AE Brain, Sequence Brain и Meta Brain.
   - Есть ли согласие между моделями или какие-то головы сигналят сильный риск отдельно.
4. Сформулируй блок "Рекомендации аналитику":
   - Каких получателей/контрагентов/каналы нужно дополнительно проверить.
   - Какие документы/подтверждения стоит запросить у клиента.
   - Нужно ли вынести кейс на углублённый анализ или достаточно мониторинга.
5. Если решение 'allow', но итоговый риск (risk_meta) выглядит существенно выше среднего (например, > 0.2–0.3), подчеркни, что кейс следует поставить на мониторинг и описать критерии дальнейшего наблюдения.
6. Цифры по суммам трактуй как значение в тенге (KZT).

Отвечай строго на русском языке, в тоне опытного аналитика банка, без лишнего жаргона. Структурируй ответ: сделай сначала краткое резюме, затем нумерованный список факторов риска, затем отдельный абзац про модели и отдельный блок с рекомендациями.
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты опытный аналитик по борьбе с мошенничеством в банке ForteBank. "
                        "Ты умеешь переводить технические признаки и выходы моделей (Fast Gate, Graph Brain, "
                        "Session Brain, AE Brain, Sequence Brain, Meta Brain) в понятные объяснения для "
                        "комплаенс-офицеров и бизнес-пользователей. Избегай технического жаргона, не упоминай SHAP, "
                        "если это не нужно, и всегда думай о том, что читатель — живой аналитик, а не дата-сайентист."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return (
            f"Ошибка при обращении к LLM: {e}\n\n"
            "Резюме кейса, политики и факторов риска для ручного анализа:\n\n"
            f"=== КЕЙС ===\n{summary}\n\n"
            f"=== ПОЛИТИКА И ПОРОГИ ===\n{policy_text}\n\n"
            f"=== ФАКТОРЫ ===\n{shap_text}"
        )


# =========================
# LLM: ЧЕРНОВИК SAR
# =========================

def generate_sar_text(
    case: Dict[str, Any],
    shap_top_features: List[Dict[str, Any]] | None = None,
    shap_top_groups: List[Dict[str, Any]] | None = None,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Черновик SAR-отчёта по кейсу.
    """
    summary = _build_case_summary(case)
    policy_text = _build_policy_summary(extra_payload)
    shap_text = _format_shap_for_prompt(shap_top_features or [], shap_top_groups or [], extra_payload)

    client = _get_client()
    if client is None:
        # простой фолбэк
        return (
            "LLM (ChatGPT API) не сконфигурирован, поэтому ниже простой шаблон SAR.\n\n"
            f"=== КЕЙС ===\n{summary}\n\n"
            f"=== ПОЛИТИКА И ПОРОГИ ===\n{policy_text}\n\n"
            f"=== ФАКТОРЫ РИСКА ===\n{shap_text}\n\n"
            "Рекомендация: провести дополнительную проверку клиента и его получателей, "
            "оценить необходимость формирования формального SAR по внутренним стандартам банка."
        )

    model = os.getenv("OPENAI_SAR_MODEL", os.getenv("OPENAI_EXPLAIN_MODEL", "gpt-4o-mini"))

    user_prompt = f"""
Нужно подготовить структурированный черновик SAR (сообщения о подозрительной активности) по транзакции.

=== КЕЙС (СЫРЫЕ ДАННЫЕ) ===
{summary}

=== ПОЛИТИКА И ПОРОГИ ===
{policy_text}

=== ФАКТОРЫ РИСКА (SHAP / REASONS) ===
{shap_text}

Сделай черновик SAR по следующей структуре (на русском языке):

1. Клиент и транзакция:
   - Кратко опиши клиента и контекст операции (тип операции, направление, сумма в тенге, дата/время).
   - Укажи, почему операция попала в зону внимания (на уровне итогового риска/решения).

2. Признаки подозрительности:
   - Список ключевых факторов риска (каждый отдельной строкой).
   - Для каждого фактора объясни, в чём аномалия или настораживающий признак (история клиента, получатель, поведение в сессии, частота операций, связи в графе и т.п.).
   - Сформулируй это так, чтобы можно было вставить в официальный SAR.

3. Возможная схема или мотив:
   - Опиши 1–2 наиболее вероятных сценария (например, обналичивание, mule-активность, обход лимитов, отмывание средств, компрометация аккаунта).
   - Если данных недостаточно, аккуратно укажи, что это гипотеза.

4. Рекомендации и дальнейшие действия:
   - Какие документы/подтверждения запросить у клиента.
   - Какие дополнительные проверки провести (по получателям, по связанным аккаунтам, по устройствам/географии).
   - Рекомендуется ли формальный SAR для регулятора или достаточно усиленного мониторинга.

5. Итоговая оценка:
   - Короткий итоговый абзац, где подводится вывод, насколько кейс серьёзный и почему.
   - Ссылаться на суммы в KZT (тенге).

Стиль:
- Строгий, деловой, без лишней воды.
- Объём: примерно 20–35 строк текста.
- Не используй технический жаргон (типа “логиты” или “SHAP”), вместо этого описывай суть признаков человеческим языком.
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты составляешь черновики SAR-отчётов для банка ForteBank. "
                        "Твоя задача — на основе технических данных и факторов риска "
                        "сформировать понятный и структурированный текст, который можно адаптировать "
                        "под формальные требования регулятора."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return (
            f"Ошибка при обращении к LLM: {e}\n\n"
            "Вот данные кейса, политика и факторы риска для ручного написания SAR:\n\n"
            f"=== КЕЙС ===\n{summary}\n\n"
            f"=== ПОЛИТИКА И ПОРОГИ ===\n{policy_text}\n\n"
            f"=== ФАКТОРЫ ===\n{shap_text}"
        )
