"""
DSS Decision Module.
Forms buy/wait/neutral recommendation based on:
  - LSTM and CatBoost forecasts
  - Ticket price history (rolling min and max)
  - Model reliability metadata (MAE, F1)

Logic (chapter 3.5 of thesis) with MAE correction:
  LSTM signal is considered significant only if |delta_rub| > lstm_mae * MAE_SIGNIFICANCE_FACTOR
  Otherwise forecast is within noise and should not influence decision

  BUY:  CatBoost class 2 (jump) with high confidence + LSTM grows significantly (or neutral)
  BUY:  Price at minimum + CatBoost sees no drop + LSTM gives no significant drop signal
  WAIT: CatBoost stable + price above minimum + LSTM drops significantly
  WAIT: CatBoost stable + price at minimum + LSTM drops significantly (NEW RULE)
  WAIT: CatBoost stable + price above minimum + LSTM neutral (noise)
  NEUTRAL: conflicting signals

Fallback:
  When ML models are missing or failed, delegates to fallback.py heuristic engine.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.config import (
    DSS_CATBOOST_CONFIDENCE_BUY,
    DSS_LSTM_GROWTH_BUY,
    JUMP_CLASS_NAMES,
)
from app.db import get_conn
from app.ml.catboost_model import predict_catboost
from app.ml.lstm_model import predict_lstm
from app.ml.training import get_active_models_info, has_active_models
from app.dss.fallback import fallback_decide, FallbackResult

logger = logging.getLogger(__name__)

MAE_SIGNIFICANCE_FACTOR = 0.5


@dataclass
class Recommendation:
    """Structure of final recommendation for the user."""
    option_name: str
    current_price: float
    recommendation: str                     # BUY | WAIT | NEUTRAL
    confidence: float
    reasoning: str
    lstm_pred_rub: Optional[float] = None
    lstm_delta_pct: Optional[float] = None
    catboost_class: Optional[int] = None
    catboost_class_name: Optional[str] = None
    catboost_proba: Optional[Dict[str, float]] = None
    price_history_min: Optional[float] = None
    price_history_max: Optional[float] = None
    price_history_mean: Optional[float] = None
    model_trust: Optional[Dict] = None
    lstm_version: Optional[str] = None
    catboost_version: Optional[str] = None
    fallback_mode: Optional[str] = None     # "heuristic" | "lstm_only" | "catboost_only" | None
    fallback_reason: Optional[str] = None   # reason for fallback activation
    price_percentile: Optional[float] = None  # position in historical range (0-100)


def _get_price_history_stats(option_name: str) -> Dict[str, float]:
    with get_conn() as c:
        row = c.execute(
            "SELECT MIN(best_price) AS min_p, MAX(best_price) AS max_p,  "
            "AVG(best_price) AS mean_p, COUNT(*) AS n  "
            "FROM observations WHERE option_name = ? ",
            (option_name,),
        ).fetchone()
    return {
        "min": float(row["min_p"]) if row["min_p"] else 0.0,
        "max": float(row["max_p"]) if row["max_p"] else 0.0,
        "mean": float(row["mean_p"]) if row["mean_p"] else 0.0,
        "n_observations": int(row["n"]),
    }


def _get_model_trust() -> Dict:
    info = get_active_models_info()
    return {
        "lstm_mae": float(info.get("lstm", {}).get("metric_value") or 0.0),
        "lstm_version": info.get("lstm", {}).get("version", "none"),
        "lstm_trained_at": info.get("lstm", {}).get("trained_at"),
        "catboost_f1": float(info.get("catboost", {}).get("metric_value") or 0.0),
        "catboost_version": info.get("catboost", {}).get("version", "none"),
        "catboost_trained_at": info.get("catboost", {}).get("trained_at"),
    }


def decide(option_name: str) -> Recommendation:
    logger.info(f"Decide for option={option_name}")

    with get_conn() as c:
        exists = c.execute(
            "SELECT 1 FROM observations WHERE option_name = ? LIMIT 1 ",
            (option_name,),
        ).fetchone()
    if not exists:
        raise ValueError(f"Ticket not found: {option_name}")

    # Check if models are available at all
    models_available = has_active_models()

    # 1. LSTM prediction (with exception handling)
    lstm_result = None
    lstm_failed = False
    try:
        lstm_result = predict_lstm(option_name)
    except Exception as e:
        lstm_failed = True
        logger.warning(f"LSTM predict failed for {option_name}: {e}")

    # 2. CatBoost prediction (with exception handling)
    cb_result = None
    cb_failed = False
    try:
        cb_result = predict_catboost(option_name)
    except Exception as e:
        cb_failed = True
        logger.warning(f"CatBoost predict failed for {option_name}: {e}")

    # 3. History + trust
    history = _get_price_history_stats(option_name)
    trust = _get_model_trust()

    # Current price
    with get_conn() as c:
        row = c.execute(
            "SELECT best_price FROM observations WHERE option_name = ?  "
            "ORDER BY observation_date DESC, time_encoded DESC LIMIT 1 ",
            (option_name,),
        ).fetchone()
    current_price = float(row["best_price"]) if row else 0.0

    # ============================================================
    # FALLBACK PATH: models missing or failed
    # ============================================================
    if not models_available or lstm_failed or cb_failed:
        # Determine fallback reason
        if not models_available:
            fallback_reason = "models_unavailable"
        elif lstm_failed and cb_failed:
            fallback_reason = "both_models_failed"
        elif lstm_failed:
            fallback_reason = "lstm_failed"
        else:
            fallback_reason = "catboost_failed"

        # Build fallback result
        fb_result = fallback_decide(
            option_name=option_name,
            history=history,
            current_price=current_price,
            n_observations=history["n_observations"],
            lstm_result=lstm_result,
            catboost_result=cb_result,
            fallback_reason=fallback_reason,
        )

        # Convert FallbackResult -> Recommendation
        rec = Recommendation(
            option_name=option_name,
            current_price=current_price,
            recommendation=fb_result.recommendation,
            confidence=fb_result.confidence,
            reasoning=fb_result.reasoning,
            lstm_pred_rub=round(lstm_result["pred_rub"], 2) if lstm_result and lstm_result.get("pred_rub") else None,
            lstm_delta_pct=round(lstm_result["delta_pct"], 3) if lstm_result and lstm_result.get("delta_pct") is not None else None,
            catboost_class=cb_result.get("predicted_class") if cb_result else None,
            catboost_class_name=(JUMP_CLASS_NAMES.get(cb_result["predicted_class"])
                                 if cb_result and cb_result.get("predicted_class") is not None else None),
            catboost_proba=cb_result.get("proba") if cb_result else None,
            price_history_min=history["min"],
            price_history_max=history["max"],
            price_history_mean=history["mean"],
            model_trust=trust,
            lstm_version=lstm_result.get("model_version") if lstm_result else None,
            catboost_version=cb_result.get("model_version") if cb_result else None,
            fallback_mode=fb_result.fallback_mode,
            fallback_reason=fb_result.fallback_reason,
            price_percentile=fb_result.price_percentile,
        )

        # Save recommendation to DB (mark as fallback)
        with get_conn() as c:
            c.execute(
                """
                INSERT INTO recommendations (
                    option_name, current_price, lstm_pred_rub, catboost_class,
                    recommendation, confidence, reasoning
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (option_name, current_price,
                 rec.lstm_pred_rub, rec.catboost_class,
                 rec.recommendation, rec.confidence, rec.reasoning),
            )
            c.commit()

        logger.info(
            f"Fallback decision for {option_name}: {rec.recommendation} "
            f"(confidence={rec.confidence}, mode={rec.fallback_mode}, reason={rec.fallback_reason})"
        )
        return rec

    # ============================================================
    # NORMAL PATH: both models available and working
    # ============================================================

    # 4. Extract values
    lstm_delta_pct = None
    lstm_delta_rub = None
    lstm_pred_rub = None
    lstm_version = None
    if lstm_result:
        lstm_delta_pct = lstm_result.get("delta_pct")
        lstm_delta_rub = lstm_result.get("delta_rub")          # ← теперь доступен из LSTM
        lstm_pred_rub = lstm_result.get("pred_rub")
        lstm_version = lstm_result.get("model_version")

    cb_class = None
    cb_proba: Optional[Dict[str, float]] = None
    cb_version = None
    cb_max_proba = 0.0
    if cb_result:
        cb_class = cb_result.get("predicted_class")
        cb_proba = cb_result.get("proba")
        cb_version = cb_result.get("model_version")
        if cb_proba and cb_class is not None:
            cb_max_proba = float(cb_proba.get(str(cb_class), 0.0))

    # ============================================================
    # 5. LSTM SIGNAL SIGNIFICANCE WITH MAE CORRECTION
    # ============================================================
    lstm_mae = trust.get("lstm_mae", 0.0)
    mae_threshold = lstm_mae * MAE_SIGNIFICANCE_FACTOR
    abs_delta_rub = abs(lstm_delta_rub) if lstm_delta_rub is not None else 0.0
    lstm_signal_significant = (lstm_delta_rub is not None) and (abs_delta_rub > mae_threshold)

    lstm_growth_signal = (
        lstm_signal_significant
        and lstm_delta_pct is not None
        and lstm_delta_pct >= DSS_LSTM_GROWTH_BUY
    )
    lstm_drop_signal = (
        lstm_signal_significant
        and lstm_delta_pct is not None
        and lstm_delta_pct <= -(DSS_LSTM_GROWTH_BUY / 2)
    )
    lstm_neutral = not lstm_signal_significant

    logger.info(
        f"LSTM signal: delta_rub={lstm_delta_rub}, mae={lstm_mae:.0f},  "
        f"threshold={mae_threshold:.0f}, significant={lstm_signal_significant},  "
        f"growth={lstm_growth_signal}, drop={lstm_drop_signal}, neutral={lstm_neutral}"
    )

    # ============================================================
    # 6. CATBOOST SIGNALS
    # ============================================================
    high_jump_signal = cb_class == 2 and cb_max_proba >= DSS_CATBOOST_CONFIDENCE_BUY
    moderate_jump_signal = cb_class == 1 and cb_max_proba >= DSS_CATBOOST_CONFIDENCE_BUY
    stable_signal = cb_class == 0 and cb_max_proba >= DSS_CATBOOST_CONFIDENCE_BUY

    price_above_min = history["min"] > 0 and current_price > history["min"] * 1.05
    price_near_min = history["min"] > 0 and current_price <= history["min"] * 1.05

    # ============================================================
    # 7. DECISION RULES
    # ============================================================
    recommendation = "NEUTRAL"
    confidence = 0.5
    reasoning_parts: List[str] = []

    # -- Rule 1: Sharp CatBoost jump + LSTM grows or neutral --
    if high_jump_signal and (lstm_growth_signal or lstm_neutral):
        recommendation = "BUY"
        confidence = min(cb_max_proba * 1.4, 0.95)
        reasoning_parts.append(
            f"CatBoost predicts sharp price jump (class 2)  "
            f"with probability {cb_max_proba:.0%}"
        )
        if lstm_growth_signal:
            reasoning_parts.append(
                f"LSTM confirms growth by {lstm_delta_pct:+.1f}%  "
                f"({lstm_delta_rub:+.0f} RUB, exceeds MAE {lstm_mae:.0f} RUB)"
            )
        else:
            reasoning_parts.append(
                f"LSTM forecast ({lstm_delta_pct:+.1f}%) within model error  "
                f"(MAE={lstm_mae:.0f} RUB) — signal is neutral"
            )
        reasoning_parts.append("Immediate purchase recommended")

    # -- Rule 2: Moderate CatBoost growth + significant LSTM growth --
    elif moderate_jump_signal and lstm_growth_signal:
        recommendation = "BUY"
        confidence = cb_max_proba * 0.9
        reasoning_parts.append(
            f"CatBoost predicts moderate growth (class 1)  "
            f"with probability {cb_max_proba:.0%}"
        )
        reasoning_parts.append(
            f"LSTM confirms growth by {lstm_delta_pct:+.1f}%  "
            f"({lstm_delta_rub:+.0f} RUB > MAE {lstm_mae:.0f} RUB)"
        )
        reasoning_parts.append("Recommended to purchase before expected price increase")

    # -- Rule 3: Price at minimum + stability + LSTM does not drop significantly --
    elif price_near_min and stable_signal and not lstm_drop_signal:
        recommendation = "BUY"
        confidence = 0.65
        reasoning_parts.append(
            f"Current price {current_price:.0f} RUB is close to historical  "
            f"minimum {history['min']:.0f} RUB"
        )
        reasoning_parts.append(
            f"CatBoost predicts stability (class 0, {cb_max_proba:.0%})"
        )
        if lstm_neutral:
            reasoning_parts.append(
                f"LSTM forecast ({lstm_delta_pct:+.1f}% =  "
                f"{lstm_delta_rub:+.0f} RUB) within MAE {lstm_mae:.0f} RUB —  "
                f"no significant growth or drop"
            )
            reasoning_parts.append(
                "Buying at minimum with no deterioration signals — optimal timing"
            )
        else:
            reasoning_parts.append(f"LSTM: change {lstm_delta_pct:+.1f}%")

    # -- Rule 4: Stable + price above minimum + LSTM drops or neutral --
    elif stable_signal and price_above_min:
        recommendation = "WAIT"
        confidence = cb_max_proba * 0.95
        reasoning_parts.append(
            f"CatBoost predicts stable price  "
            f"with probability {cb_max_proba:.0%}"
        )
        reasoning_parts.append(
            f"Current price {current_price:.0f} RUB is {((current_price / history['min']) - 1) * 100:.1f}%  "
            f"above historical minimum {history['min']:.0f} RUB"
        )
        if lstm_drop_signal:
            reasoning_parts.append(
                f"LSTM predicts significant drop by {lstm_delta_pct:+.1f}%  "
                f"({lstm_delta_rub:+.0f} RUB > MAE {lstm_mae:.0f} RUB)"
            )
        elif lstm_neutral:
            reasoning_parts.append(
                f"LSTM forecast ({lstm_delta_pct:+.1f}%) within error margin —  "
                f"no reversal signal"
            )
        reasoning_parts.append("Recommended to wait for price decrease")

    # -- Rule 5: Price at minimum + LSTM drops significantly (NEW!) --
    elif price_near_min and lstm_drop_signal:
        recommendation = "WAIT"
        confidence = 0.70
        reasoning_parts.append(
            f"Current price {current_price:.0f} RUB is close to minimum {history['min']:.0f} RUB"
        )
        reasoning_parts.append(
            f"LSTM predicts significant drop by {lstm_delta_pct:+.1f}%  "
            f"({lstm_delta_rub:+.0f} RUB > MAE {lstm_mae:.0f} RUB)"
        )
        if stable_signal:
            reasoning_parts.append(
                f"CatBoost predicts stability ({cb_max_proba:.0%}),  "
                f"but LSTM sees potential decline"
            )
        reasoning_parts.append(
            "Recommended to wait even at minimum price — local dip possible"
        )

    # -- Rule 6: NEUTRAL — conflicting or weak signals --
    else:
        recommendation = "NEUTRAL"
        confidence = 0.5
        reasoning_parts.append("Model signals are conflicting or insufficiently confident")
        if cb_class is not None:
            class_name = JUMP_CLASS_NAMES.get(cb_class, str(cb_class))
            reasoning_parts.append(
                f"CatBoost: {class_name} with probability {cb_max_proba:.0%}"
            )
        if lstm_delta_pct is not None:
            sig_note = "significant" if lstm_signal_significant else f"within MAE {lstm_mae:.0f} RUB"
            reasoning_parts.append(
                f"LSTM: {lstm_delta_pct:+.1f}% ({sig_note})"
            )
        reasoning_parts.append("Final decision is up to the user")

    reasoning = ".  ".join(reasoning_parts) + "."

    # Compute price percentile for normal path too
    if history["max"] > history["min"]:
        price_percentile = (current_price - history["min"]) / (history["max"] - history["min"]) * 100.0
    else:
        price_percentile = 50.0

    rec = Recommendation(
        option_name=option_name,
        current_price=current_price,
        recommendation=recommendation,
        confidence=round(float(confidence), 3),
        reasoning=reasoning,
        lstm_pred_rub=round(lstm_pred_rub, 2) if lstm_pred_rub is not None else None,
        lstm_delta_pct=round(lstm_delta_pct, 3) if lstm_delta_pct is not None else None,
        catboost_class=cb_class,
        catboost_class_name=JUMP_CLASS_NAMES.get(cb_class) if cb_class is not None else None,
        catboost_proba=cb_proba,
        price_history_min=history["min"],
        price_history_max=history["max"],
        price_history_mean=history["mean"],
        model_trust=trust,
        lstm_version=lstm_version,
        catboost_version=cb_version,
        fallback_mode=None,
        fallback_reason=None,
        price_percentile=round(price_percentile, 1),
    )

    with get_conn() as c:
        c.execute(
            """
            INSERT INTO recommendations (
                option_name, current_price, lstm_pred_rub, catboost_class,
                recommendation, confidence, reasoning
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (option_name, current_price, lstm_pred_rub, cb_class,
             recommendation, confidence, reasoning),
        )
        c.commit()

    return rec
