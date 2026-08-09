"""
Fallback-модуль принятия решений при отсутствии ML-моделей.

Активируется когда:
  - Нет активных моделей в системе (первый запуск, сбой обучения)
  - LSTM или CatBoost выбросили исключение при инференсе
  - Модели есть, но их метрики ниже минимально допустимых порогов

Логика:
  1. Чисто эвристическая — на основе положения цены в историческом диапазоне
  2. Гибридная — доступна одна модель, используем её с пониженной уверенностью

Интерфейс:
  fallback_decide(option_name, history, current_price,
                  lstm_result, catboost_result, fallback_reason)
    → FallbackResult (структура аналогичная Recommendation)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.config import (
    FALLBACK_CONFIDENCE_BUY_MIN,
    FALLBACK_CONFIDENCE_WAIT_MIN,
    FALLBACK_NEUTRAL_ZONE_PCT,
    FALLBACK_PRICE_MIN_PCT,
    FALLBACK_PRICE_MAX_PCT,
)

logger = logging.getLogger(__name__)


@dataclass
class FallbackResult:
    """Результат fallback-принятия решения."""
    option_name: str
    current_price: float
    recommendation: str          # BUY | WAIT | NEUTRAL
    confidence: float
    reasoning: str
    fallback_mode: str           # "heuristic" | "lstm_only" | "catboost_only"
    fallback_reason: str         # почему активирован fallback
    price_history_min: Optional[float] = None
    price_history_max: Optional[float] = None
    price_history_mean: Optional[float] = None
    price_percentile: Optional[float] = None
    lstm_delta_pct: Optional[float] = None
    catboost_class: Optional[int] = None


def _compute_price_percentile(current: float, hist_min: float, hist_max: float) -> float:
    """Вычисляет процентиль текущей цены в историческом диапазоне. 0%=min, 100%=max."""
    if hist_max <= hist_min:
        return 50.0
    pct = (current - hist_min) / (hist_max - hist_min) * 100.0
    return max(0.0, min(100.0, pct))


def _heuristic_decision(
    current_price: float,
    history: Dict[str, float],
    n_observations: int,
) -> tuple[str, float, str]:
    """
    Чисто эвристическое решение на основе положения цены в истории.

    Правила (по приоритету):
      1. Цена у минимума (<= FALLBACK_PRICE_MIN_PCT%)  -> BUY
      2. Цена у максимума (>= FALLBACK_PRICE_MAX_PCT%) -> WAIT
      3. Наблюдений < 5                                 -> NEUTRAL (нет истории)
      4. В neutral-зоне вокруг средней                  -> NEUTRAL
      5. Цена ниже средней                              -> BUY (слабый)
      6. Цена выше средней                              -> WAIT (слабый)
    """
    hist_min = history.get("min", 0.0)
    hist_max = history.get("max", 0.0)
    hist_mean = history.get("mean", 0.0)

    percentile = _compute_price_percentile(current_price, hist_min, hist_max)

    parts: List[str] = []
    parts.append(
        f"ML-модели недоступны. Решение на основе истории цен "
        f"({n_observations} наблюдений, диапазон {hist_min:.0f}-{hist_max:.0f} RUB)"
    )

    # Rule 1: Price near minimum -> BUY
    if percentile <= FALLBACK_PRICE_MIN_PCT:
        confidence = max(FALLBACK_CONFIDENCE_BUY_MIN, 0.75 - percentile * 0.01)
        parts.append(
            f"Current price {current_price:.0f} RUB is close to historical minimum "
            f"{hist_min:.0f} RUB ({percentile:.1f}% of range)"
        )
        parts.append("Recommended to buy — price is in favorable zone")
        return "BUY", confidence, ". ".join(parts) + "."

    # Rule 2: Price near maximum -> WAIT
    if percentile >= FALLBACK_PRICE_MAX_PCT:
        confidence = max(FALLBACK_CONFIDENCE_WAIT_MIN, 0.55 + (percentile - 80) * 0.01)
        parts.append(
            f"Current price {current_price:.0f} RUB is close to historical maximum "
            f"{hist_max:.0f} RUB ({percentile:.1f}% of range)"
        )
        parts.append("Recommended to wait — price may decrease")
        return "WAIT", confidence, ". ".join(parts) + "."

    # Rule 3: Not enough observations
    if n_observations < 5:
        parts.append(
            f"Not enough observations ({n_observations}) for reliable conclusion. "
            f"Data collection continues."
        )
        return "NEUTRAL", 0.4, ". ".join(parts) + "."

    # Rule 4: Neutral zone around mean
    if hist_mean > 0:
        neutral_lower = hist_mean * (1.0 - FALLBACK_NEUTRAL_ZONE_PCT / 100.0)
        neutral_upper = hist_mean * (1.0 + FALLBACK_NEUTRAL_ZONE_PCT / 100.0)
        if neutral_lower <= current_price <= neutral_upper:
            parts.append(
                f"Price {current_price:.0f} RUB is in neutral zone "
                f"({neutral_lower:.0f}-{neutral_upper:.0f} RUB) relative to mean {hist_mean:.0f} RUB"
            )
            parts.append("No clear signal — decision is up to the user")
            return "NEUTRAL", 0.45, ". ".join(parts) + "."

    # Rules 5-6: Below or above mean
    if current_price < hist_mean:
        parts.append(
            f"Price {current_price:.0f} RUB is below mean {hist_mean:.0f} RUB "
            f"({percentile:.1f}% of range)"
        )
        parts.append("Weak buy signal")
        return "BUY", 0.55, ". ".join(parts) + "."
    else:
        parts.append(
            f"Price {current_price:.0f} RUB is above mean {hist_mean:.0f} RUB "
            f"({percentile:.1f}% of range)"
        )
        parts.append("Weak wait signal")
        return "WAIT", 0.55, ". ".join(parts) + "."


def _single_model_decision(
    current_price: float,
    history: Dict[str, float],
    n_observations: int,
    lstm_result: Optional[Dict] = None,
    catboost_result: Optional[Dict] = None,
) -> tuple[str, float, str, str]:
    """
    Hybrid fallback: one model is available.
    Use its signals with reduced confidence (~25% penalty).
    """
    hist_min = history.get("min", 0.0)
    hist_mean = history.get("mean", 0.0)

    price_above_min = hist_min > 0 and current_price > hist_min * 1.05
    price_near_min = hist_min > 0 and current_price <= hist_min * 1.05

    # === LSTM only ===
    if lstm_result is not None and catboost_result is None:
        lstm_delta_pct = lstm_result.get("delta_pct", 0.0)
        pred_rub = lstm_result.get("pred_rub", current_price)
        lstm_version = lstm_result.get("model_version", "unknown")

        parts: List[str] = []
        parts.append(
            f"CatBoost unavailable. Decision based on LSTM v{lstm_version} "
            f"(-confidence 25% due to missing second model)"
        )

        if lstm_delta_pct is not None:
            if lstm_delta_pct >= 0.5:  # Growth
                rec = "BUY"
                conf = 0.55
                parts.append(
                    f"LSTM predicts growth by {lstm_delta_pct:+.1f}% "
                    f"(to ~{pred_rub:.0f} RUB)"
                )
                parts.append("Weak buy recommendation — single model only")
            elif lstm_delta_pct <= -0.5:  # Decline
                rec = "WAIT"
                conf = 0.55
                parts.append(
                    f"LSTM predicts decline by {lstm_delta_pct:+.1f}% "
                    f"(to ~{pred_rub:.0f} RUB)"
                )
                if price_above_min:
                    parts.append("Price is above minimum — recommended to wait")
                elif price_near_min:
                    parts.append("Price at minimum but LSTM sees decline — may wait")
                else:
                    parts.append("Recommended to wait for price drop")
            else:  # Neutral
                rec = "NEUTRAL"
                conf = 0.45
                parts.append(f"LSTM forecast is neutral ({lstm_delta_pct:+.1f}%)")
                parts.append("Insufficient data for confident decision")
        else:
            rec = "NEUTRAL"
            conf = 0.4
            parts.append("LSTM did not provide numerical forecast")

        return rec, conf, ". ".join(parts) + ".", "lstm_only"

    # === CatBoost only ===
    if catboost_result is not None and lstm_result is None:
        cb_class = catboost_result.get("predicted_class")
        cb_proba = catboost_result.get("proba", {})
        cb_version = catboost_result.get("model_version", "unknown")
        cb_max_proba = float(cb_proba.get(str(cb_class), 0.0)) if cb_proba and cb_class is not None else 0.0

        parts: List[str] = []
        parts.append(
            f"LSTM unavailable. Decision based on CatBoost v{cb_version} "
            f"(class {cb_class}, -confidence 25% due to missing second model)"
        )

        if cb_class == 2 and cb_max_proba >= 0.35:
            rec = "BUY"
            conf = min(cb_max_proba * 0.75, 0.70)
            parts.append(
                f"CatBoost predicts sharp jump (class 2) with probability {cb_max_proba:.0%}"
            )
            parts.append("Recommended to buy, but confidence is reduced — single model only")
        elif cb_class == 1 and cb_max_proba >= 0.35:
            rec = "BUY"
            conf = cb_max_proba * 0.65
            parts.append(
                f"CatBoost predicts moderate growth (class 1) with probability {cb_max_proba:.0%}"
            )
            parts.append("Weak buy recommendation")
        elif cb_class == 0 and cb_max_proba >= 0.35:
            rec = "WAIT"
            conf = cb_max_proba * 0.70
            parts.append(
                f"CatBoost predicts stability (class 0) with probability {cb_max_proba:.0%}"
            )
            if price_above_min:
                parts.append(f"Price {current_price:.0f} RUB above minimum {hist_min:.0f} RUB — can wait")
            elif price_near_min:
                rec = "BUY"
                conf = 0.55
                parts.append(
                    f"Price {current_price:.0f} RUB close to minimum {hist_min:.0f} RUB — "
                    f"buying at minimum is optimal"
                )
            else:
                parts.append("Recommended to wait")
        else:
            rec = "NEUTRAL"
            conf = 0.4
            parts.append(f"CatBoost gave weak or ambiguous signal (class {cb_class})")
            parts.append("Insufficient data for confident decision")

        return rec, conf, ". ".join(parts) + ".", "catboost_only"

    # Fallback safeguard
    rec, conf, reasoning = _heuristic_decision(current_price, history, n_observations)
    return rec, conf, reasoning, "heuristic"


def fallback_decide(
    option_name: str,
    history: Dict[str, float],
    current_price: float,
    n_observations: int,
    lstm_result: Optional[Dict] = None,
    catboost_result: Optional[Dict] = None,
    fallback_reason: str = "models_unavailable",
) -> FallbackResult:
    """
    Main fallback entry point.

    Parameters:
      option_name      — route key
      history          — {min, max, mean} from _get_price_history_stats
      current_price    — current ticket price
      n_observations   — number of observations in history
      lstm_result      — predict_lstm() result (None if unavailable)
      catboost_result  — predict_catboost() result (None if unavailable)
      fallback_reason  — "models_unavailable" | "lstm_failed" | "catboost_failed" | "metrics_too_low"

    Returns:
      FallbackResult with recommendation, confidence, reasoning
    """
    logger.info(
        f"Fallback activated for {option_name}: reason={fallback_reason}, "
        f"lstm={'ok' if lstm_result else 'none'}, "
        f"catboost={'ok' if catboost_result else 'none'}"
    )

    if lstm_result is None and catboost_result is None:
        rec, conf, reasoning = _heuristic_decision(current_price, history, n_observations)
        mode = "heuristic"
    elif lstm_result is not None and catboost_result is not None:
        logger.warning("fallback_decide called with both models available, using heuristics anyway")
        rec, conf, reasoning = _heuristic_decision(current_price, history, n_observations)
        mode = "heuristic"
    else:
        rec, conf, reasoning, mode = _single_model_decision(
            current_price, history, n_observations,
            lstm_result, catboost_result,
        )

    percentile = _compute_price_percentile(
        current_price,
        history.get("min", 0.0),
        history.get("max", 0.0),
    )

    lstm_delta = lstm_result.get("delta_pct") if lstm_result else None
    cb_class = catboost_result.get("predicted_class") if catboost_result else None

    result = FallbackResult(
        option_name=option_name,
        current_price=current_price,
        recommendation=rec,
        confidence=round(float(conf), 3),
        reasoning=reasoning,
        fallback_mode=mode,
        fallback_reason=fallback_reason,
        price_history_min=history.get("min"),
        price_history_max=history.get("max"),
        price_history_mean=history.get("mean"),
        price_percentile=round(percentile, 1),
        lstm_delta_pct=lstm_delta,
        catboost_class=cb_class,
    )

    logger.info(
        f"Fallback result for {option_name}: {rec} (confidence={conf:.3f}, "
        f"mode={mode}, percentile={percentile:.1f}%)"
    )
    return result
