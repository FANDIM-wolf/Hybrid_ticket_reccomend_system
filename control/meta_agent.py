"""
Мета-агент модуля управления.

Реализует двухголовую схему из главы 2 ВКР:

  Голова 1 — «Когда переобучать»: A_1 = {retrain, skip}, награда Reward_1 = ΔMetric − λ·C
  Голова 2 — «Как переобучать»: 8 преднастроенных конфигураций (arm pruning),
             награда Reward_2 = ΔMetric (чистое улучшение после retrain)

Обе головы используют LinUCB поверх одного и того же вектора состояния среды.
Голова 2 активируется только когда Голова 1 выбрала retrain.
Между ними есть важное различие: Голова 1 всегда обновляется после каждого решения,
Голова 2 — только после реально выполненных retrain.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import RETRAIN_COST, RETRAIN_COST_WEIGHT
from app.control.drift_detector import DDMDetector, PageHinkleyDetector
from app.control.linucb import LinUCBBandit
from app.control.observation_buffer import ObservationBuffer
from app.db import get_conn
from app.ml.training import TrainConfig, get_active_models_info, train_all

logger = logging.getLogger(__name__)


# ============================================================
# ПРОСТРАНСТВО ДЕЙСТВИЙ ГОЛОВЫ 2 — 8 пресетов (arm pruning)
# ============================================================
HEAD2_PRESETS = {
    # Стандартный ансамбль 15 моделей, balanced веса
    "preset_default": TrainConfig(
        history_window_days=None, lookback=14,
        class_weights_mode="balanced", lstm_learning_rate=5e-4,
        n_ensemble=15, synthetic_weight=0.4,
    ),
    # Свежее окно — реакция на дрейф, ансамбль поменьше
    "preset_recent": TrainConfig(
        history_window_days=30, lookback=14,
        class_weights_mode="balanced", lstm_learning_rate=5e-4,
        n_ensemble=10, synthetic_weight=0.4,
    ),
    # Длинный lookback LSTM — больше контекст
    "preset_long_lookback": TrainConfig(
        history_window_days=None, lookback=20,
        class_weights_mode="balanced", lstm_learning_rate=5e-4,
        n_ensemble=15, synthetic_weight=0.4,
    ),
    # Короткий lookback — быстрая реакция
    "preset_short_lookback": TrainConfig(
        history_window_days=None, lookback=8,
        class_weights_mode="balanced", lstm_learning_rate=5e-4,
        n_ensemble=15, synthetic_weight=0.4,
    ),
    # Усиленные веса на скачки + большой ансамбль
    "preset_boost_jumps": TrainConfig(
        history_window_days=None, lookback=14,
        class_weights_mode="boost_jump", lstm_learning_rate=5e-4,
        n_ensemble=20, synthetic_weight=0.4,
    ),
    # Низкий LSTM lr — тонкая подстройка
    "preset_low_lr": TrainConfig(
        history_window_days=None, lookback=14,
        class_weights_mode="balanced", lstm_learning_rate=2e-4,
        n_ensemble=15, synthetic_weight=0.4,
    ),
    # Агрессивный: короткое окно + быстрый lstm + малый ансамбль
    "preset_aggressive": TrainConfig(
        history_window_days=30, lookback=10,
        class_weights_mode="balanced", lstm_learning_rate=1e-3,
        n_ensemble=8, synthetic_weight=0.5,
    ),
    # Стабильный: длинное окно + большой ансамбль + малый lr
    "preset_stable": TrainConfig(
        history_window_days=60, lookback=14,
        class_weights_mode="balanced", lstm_learning_rate=2e-4,
        n_ensemble=25, synthetic_weight=0.3,
    ),
}

HEAD1_ARMS = ["retrain", "skip"]

# Вектор состояния среды S_t = ⟨PH_t, p_t, Δ_t, N_t, F1_t, MAE_t, bias⟩
# bias = 1.0 для корректной работы линейной модели LinUCB
STATE_DIM = 7


# ============================================================
# CONTROL LOG
# ============================================================
def _log_event(
    event_type: str,
    ph_statistic: Optional[float] = None,
    ddm_error_rate: Optional[float] = None,
    state_vector: Optional[np.ndarray] = None,
    action_chosen: Optional[str] = None,
    expected_reward: Optional[float] = None,
    message: Optional[str] = None,
) -> None:
    state_json = json.dumps(state_vector.tolist()) if state_vector is not None else None
    with get_conn() as c:
        c.execute(
            """
            INSERT INTO control_logs (
                event_type, ph_statistic, ddm_error_rate, state_vector_json,
                action_chosen, expected_reward, message
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (event_type, ph_statistic, ddm_error_rate, state_json,
             action_chosen, expected_reward, message),
        )
        c.commit()


# ============================================================
# СТРОИМ ВЕКТОР СОСТОЯНИЯ
# ============================================================
def build_state_vector(
    ph_stat: float,
    ddm_error_rate: float,
    drift_magnitude: float,
    n_new_observations: int,
    current_f1: float,
    current_mae: float,
) -> np.ndarray:
    """
    Формирует вектор состояния среды S_t из главы 2 ВКР.
    Нормализованы до разумных диапазонов, чтобы линейный LinUCB работал корректно.
    """
    return np.array([
        float(ph_stat),                    # PH_t
        float(ddm_error_rate),             # p_t
        float(drift_magnitude),            # Δ_t
        min(float(n_new_observations) / 1000.0, 1.0),  # N_t, нормализовано
        float(current_f1),                 # F1_t ∈ [0, 1]
        min(float(current_mae) / 10000.0, 1.0),        # MAE_t, нормализовано
        1.0,                               # bias
    ], dtype=np.float64)


# ============================================================
# META-AGENT
# ============================================================
class MetaAgent:
    """
    Мета-агент с двумя головами. Центральная точка координации:
      observation_buffer → drift_detector → state_vector → LinUCB → retrain.
    """

    def __init__(self):
        self.buffer = ObservationBuffer()
        self.ph = PageHinkleyDetector()
        self.ddm = DDMDetector()
        self.head1 = LinUCBBandit(
            head_id="head1",
            arm_ids=HEAD1_ARMS,
            context_dim=STATE_DIM,
        )
        self.head2 = LinUCBBandit(
            head_id="head2",
            arm_ids=list(HEAD2_PRESETS.keys()),
            context_dim=STATE_DIM,
        )

    # ============================================================
    # МОНИТОРИНГ
    # ============================================================
    def run_drift_check(self, residual_limit: int = 200) -> Dict:
        """
        Читает остатки из буфера, прогоняет детекторы и возвращает характеристики дрейфа.
        Не принимает решений — только диагностика.
        """
        residuals = self.buffer.get_lstm_residuals(limit=residual_limit)
        correctness = self.buffer.get_catboost_correctness(limit=residual_limit)

        ph_alarm = False
        if residuals:
            ph_alarm = self.ph.fit_batch(residuals)
        ph_stat = self.ph.ph_statistic

        ddm_level = "stable"
        if correctness:
            ddm_level = self.ddm.fit_batch(correctness)
        ddm_error = self.ddm.error_rate

        drift_magnitude = abs(ph_stat) if residuals else 0.0

        result = {
            "ph_alarm": ph_alarm,
            "ph_statistic": ph_stat,
            "ddm_level": ddm_level,
            "ddm_error_rate": ddm_error,
            "drift_magnitude": drift_magnitude,
            "n_residuals": len(residuals),
            "n_correctness": len(correctness),
        }
        _log_event(
            "drift_check",
            ph_statistic=ph_stat,
            ddm_error_rate=ddm_error,
            message=json.dumps(result),
        )
        return result

    # ============================================================
    # СБОР СОСТОЯНИЯ
    # ============================================================
    def build_current_state(self) -> np.ndarray:
        """Собирает вектор состояния для передачи обеим головам."""
        drift = self.run_drift_check()

        # Текущие метрики моделей
        info = get_active_models_info()
        current_f1 = 0.0
        current_mae = 0.0
        if "catboost" in info:
            current_f1 = float(info["catboost"]["metric_value"])
        if "lstm" in info:
            current_mae = float(info["lstm"]["metric_value"])

        # Количество новых наблюдений в буфере
        n_new = self.buffer.count()

        return build_state_vector(
            ph_stat=drift["ph_statistic"],
            ddm_error_rate=drift["ddm_error_rate"],
            drift_magnitude=drift["drift_magnitude"],
            n_new_observations=n_new,
            current_f1=current_f1,
            current_mae=current_mae,
        )

    # ============================================================
    # HEAD 1: КОГДА ПЕРЕОБУЧАТЬ
    # ============================================================
    def decide_retrain(self, state: Optional[np.ndarray] = None) -> Dict:
        """
        Вызов Головы 1. Возвращает решение и ожидаемые награды.
        Не запускает retrain сам — только возвращает решение.
        """
        if state is None:
            state = self.build_current_state()

        decision = self.head1.select_arm(state)
        chosen = decision["chosen_arm"]

        _log_event(
            "head1_decision",
            state_vector=state,
            action_chosen=chosen,
            expected_reward=decision["scores"][chosen],
            message=json.dumps({
                "scores": decision["scores"],
                "means": decision["mean_rewards"],
                "bonuses": decision["bonuses"],
            }),
        )
        return {"state": state, "decision": decision}

    # ============================================================
    # HEAD 2: КАК ПЕРЕОБУЧАТЬ
    # ============================================================
    def decide_config(self, state: np.ndarray) -> Tuple[str, TrainConfig, Dict]:
        """Вызов Головы 2. Выбирает preset и возвращает TrainConfig."""
        decision = self.head2.select_arm(state)
        chosen = decision["chosen_arm"]
        cfg = HEAD2_PRESETS[chosen]

        _log_event(
            "head2_decision",
            state_vector=state,
            action_chosen=chosen,
            expected_reward=decision["scores"][chosen],
            message=json.dumps({
                "preset_config": cfg.to_dict(),
                "scores": decision["scores"],
            }),
        )
        return chosen, cfg, decision

    # ============================================================
    # ПОЛНЫЙ ЦИКЛ: МОНИТОРИНГ → РЕШЕНИЕ → (RETRAIN)
    # ============================================================
    def tick(self, force_retrain: bool = False) -> Dict:
        """
        Один "такт" работы мета-агента:
          1. Снимает состояние среды
          2. Голова 1 решает retrain / skip
          3. Если retrain — Голова 2 выбирает пресет, запускается train_all()
          4. Обе головы обновляются наградами
          5. Результат возвращается и логируется
        """
        logger.info("=== MetaAgent tick ===")
        state = self.build_current_state()

        if force_retrain:
            h1_chosen = "retrain"
            h1_decision = {"chosen_arm": "retrain", "scores": {"retrain": 0.0, "skip": 0.0},
                           "mean_rewards": {"retrain": 0.0, "skip": 0.0},
                           "bonuses": {"retrain": 0.0, "skip": 0.0}}
            logger.info("Force retrain requested")
        else:
            h1_result = self.decide_retrain(state)
            h1_decision = h1_result["decision"]
            h1_chosen = h1_decision["chosen_arm"]

        result = {
            "state_vector": state.tolist(),
            "head1": {"chosen": h1_chosen, "scores": h1_decision["scores"]},
            "head2": None,
            "retrain_result": None,
        }

        if h1_chosen == "skip":
            # Награда Головы 1 за skip = прирост метрики (=0) минус стоимость (=0)
            reward1 = 0.0
            self.head1.update("skip", state, reward1)
            _log_event(
                "head1_update",
                action_chosen="skip",
                expected_reward=reward1,
                message="skip decision, no retrain performed",
            )
            return result

        # === retrain ===
        h2_chosen, cfg, h2_decision = self.decide_config(state)
        result["head2"] = {"chosen": h2_chosen, "config": cfg.to_dict(),
                           "scores": h2_decision["scores"]}

        info_before = get_active_models_info()
        lstm_mae_before = float(info_before.get("lstm", {}).get("metric_value") or 0.0)
        cb_f1_before = float(info_before.get("catboost", {}).get("metric_value") or 0.0)

        _log_event(
            "retrain_start",
            state_vector=state,
            action_chosen=h2_chosen,
            message=json.dumps(cfg.to_dict()),
        )
        t0 = time.time()
        try:
            train_result = train_all(cfg)
        except Exception as e:
            logger.error(f"train_all failed: {e}")
            _log_event("retrain_end", message=f"FAILED: {e}")
            result["retrain_result"] = {"error": str(e)}
            return result
        duration = time.time() - t0

        info_after = get_active_models_info()
        lstm_mae_after = float(info_after.get("lstm", {}).get("metric_value") or 0.0)
        cb_f1_after = float(info_after.get("catboost", {}).get("metric_value") or 0.0)

        # ============================================================
        # НАГРАДЫ
        # ============================================================
        # Delta_metric = улучшение Macro F1 CatBoost + снижение MAE LSTM (нормализованное)
        # используем F1 как основной показатель, т.к. он безразмерный и стабильный
        delta_f1 = cb_f1_after - cb_f1_before
        delta_mae_norm = (lstm_mae_before - lstm_mae_after) / max(lstm_mae_before, 1.0)
        delta_metric = delta_f1 + 0.5 * delta_mae_norm

        reward2 = delta_metric
        reward1 = delta_metric - RETRAIN_COST_WEIGHT * RETRAIN_COST

        # Обновление головы 1 — за принятое решение retrain
        self.head1.update("retrain", state, reward1)
        self.head2.update(h2_chosen, state, reward2)

        # Журналирование retrain
        with get_conn() as c:
            c.execute(
                """
                INSERT INTO retrain_events (
                    trigger_reason, config_json,
                    lstm_mae_before, lstm_mae_after,
                    catboost_f1_before, catboost_f1_after,
                    reward_head1, reward_head2, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("manual" if force_retrain else "drift",
                 json.dumps(cfg.to_dict()),
                 lstm_mae_before, lstm_mae_after,
                 cb_f1_before, cb_f1_after,
                 reward1, reward2, duration),
            )
            c.commit()

        _log_event(
            "retrain_end",
            action_chosen=h2_chosen,
            expected_reward=reward1,
            message=json.dumps({
                "lstm_mae_before": lstm_mae_before,
                "lstm_mae_after": lstm_mae_after,
                "catboost_f1_before": cb_f1_before,
                "catboost_f1_after": cb_f1_after,
                "reward1": reward1,
                "reward2": reward2,
                "duration_seconds": duration,
            }),
        )

        # Согласно главе 2.4 ВКР: сбрасываем внутренние переменные детекторов после retrain
        self.ph.reset()
        self.ddm.reset()

        result["retrain_result"] = {
            "duration_seconds": duration,
            "lstm_mae_before": lstm_mae_before,
            "lstm_mae_after": lstm_mae_after,
            "catboost_f1_before": cb_f1_before,
            "catboost_f1_after": cb_f1_after,
            "reward_head1": reward1,
            "reward_head2": reward2,
        }
        return result

    # ============================================================
    # ДИАГНОСТИКА
    # ============================================================
    def get_state_summary(self) -> Dict:
        """Агрегированная сводка для эндпоинта /control/state."""
        return {
            "head1_summary": self.head1.summary(),
            "head2_summary": self.head2.summary(),
            "buffer_size": self.buffer.count(),
            "page_hinkley_statistic": self.ph.ph_statistic,
            "ddm_level": self.ddm.state.level,
            "ddm_error_rate": self.ddm.error_rate,
        }
