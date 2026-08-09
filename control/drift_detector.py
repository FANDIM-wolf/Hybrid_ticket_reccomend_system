"""
Детекторы концепт-дрейфа.

Правильная реализация Page-Hinkley (двусторонний):
  μ_0 — фиксированное среднее, оценённое на первых min_samples наблюдениях
  m^+_t = m^+_{t-1} + (x_t - μ_0) - δ       (положительный сдвиг)
  m^-_t = m^-_{t-1} - (x_t - μ_0) - δ       (отрицательный сдвиг)
  PH^+_t = m^+_t - min_{i≤t}(m^+_i)
  PH^-_t = max_{i≤t}(m^-_i) - m^-_t
  alarm ← max(PH^+_t, PH^-_t) > λ

DDM — стандартный Drift Detection Method для бинарного потока ошибок.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

from app.config import DDM_DRIFT_LEVEL, DDM_WARNING_LEVEL, PH_DELTA, PH_LAMBDA

logger = logging.getLogger(__name__)


# ============================================================
# PAGE-HINKLEY TEST (двусторонний, с фиксированным μ_0)
# ============================================================
@dataclass
class PageHinkleyState:
    """Внутреннее состояние теста Пейджа-Хинкли."""
    n: int = 0
    mu_0: float = 0.0                 # фиксированное среднее (до дрейфа)
    cum_sum_pos: float = 0.0          # m^+_t — положительный сдвиг
    min_sum_pos: float = 0.0          # min(m^+_i)
    cum_sum_neg: float = 0.0          # m^-_t — отрицательный сдвиг
    max_sum_neg: float = 0.0          # max(m^-_i)
    last_ph_statistic: float = 0.0    # max(PH^+, PH^-)
    is_initialized: bool = False


class PageHinkleyDetector:
    """
    Двусторонний тест Пейджа-Хинкли для обнаружения сдвига среднего остатков.

    Формулы (классическая версия):
      1. Инициализация: μ_0 = mean(x_1, ..., x_N) на первых min_samples.
         После оценки μ_0 фиксируется и не обновляется.
      2. Основная фаза (t > N):
         m^+_t = Σ_{i=N+1}^{t} [(x_i - μ_0) - δ]
         m^-_t = Σ_{i=N+1}^{t} [-(x_i - μ_0) - δ]
         PH^+_t = m^+_t - min_{N+1≤i≤t}(m^+_i)
         PH^-_t = max_{N+1≤i≤t}(m^-_i) - m^-_t
         PH_t   = max(PH^+_t, PH^-_t)
         alarm  ← PH_t > λ

    Параметры:
      delta   — минимальное смещение, которое игнорируется (порог чувствительности)
      lambda_ — порог срабатывания тревоги
      min_samples — сколько наблюдений нужно для оценки μ_0 (≥ 2)
    """

    def __init__(
        self,
        delta: float = PH_DELTA,
        lambda_: float = PH_LAMBDA,
        min_samples: int = 30,
    ):
        self.delta = delta
        self.lambda_ = lambda_
        self.min_samples = max(min_samples, 2)
        self.state = PageHinkleyState()

    def reset(self) -> None:
        self.state = PageHinkleyState()

    def add(self, residual: float) -> bool:
        """
        Добавляет один остаток. Возвращает True при срабатывании тревоги.
        """
        s = self.state
        s.n += 1

        # --- Фаза инициализации: накапливаем μ_0 ---
        if not s.is_initialized:
            if s.n == 1:
                s.mu_0 = residual
            else:
                # Онлайн-среднее (Welford) только для инициализации
                s.mu_0 += (residual - s.mu_0) / s.n

            if s.n >= self.min_samples:
                s.is_initialized = True
                # Сбрасываем аккумуляторы перед основной фазой
                s.cum_sum_pos = 0.0
                s.min_sum_pos = 0.0
                s.cum_sum_neg = 0.0
                s.max_sum_neg = 0.0
                logger.info(
                    f"Page-Hinkley initialized: μ_0={s.mu_0:.4f} "
                    f"from {self.min_samples} samples, δ={self.delta}, λ={self.lambda_}"
                )
            return False

        # --- Основная фаза: классический Page-Hinkley с фиксированным μ_0 ---
        centered = residual - s.mu_0

        # Положительный сдвиг (среднее увеличилось)
        s.cum_sum_pos += centered - self.delta
        if s.cum_sum_pos < s.min_sum_pos:
            s.min_sum_pos = s.cum_sum_pos
        ph_pos = s.cum_sum_pos - s.min_sum_pos

        # Отрицательный сдвиг (среднее уменьшилось)
        s.cum_sum_neg += -centered - self.delta
        if s.cum_sum_neg > s.max_sum_neg:
            s.max_sum_neg = s.cum_sum_neg
        ph_neg = s.max_sum_neg - s.cum_sum_neg

        s.last_ph_statistic = max(ph_pos, ph_neg)
        alarm = s.last_ph_statistic > self.lambda_

        if alarm:
            logger.warning(
                f"Page-Hinkley ALARM: PH={s.last_ph_statistic:.3f} > λ={self.lambda_} "
                f"(μ_0={s.mu_0:.4f}, residual={residual:.4f})"
            )

        return alarm

    def fit_batch(self, residuals: List[float]) -> bool:
        """Применить детектор к батчу. Возвращает True, если сработал хотя бы раз."""
        self.reset()
        alarmed = False
        for r in residuals:
            if self.add(r):
                alarmed = True
        return alarmed

    @property
    def ph_statistic(self) -> float:
        return self.state.last_ph_statistic


# ============================================================
# DDM TEST (для классификатора CatBoost)
# ============================================================
@dataclass
class DDMState:
    """Внутреннее состояние метода DDM."""
    n: int = 0
    n_errors: int = 0
    p_min: float = float("inf")
    s_min: float = float("inf")
    last_p: float = 0.0
    last_s: float = 0.0
    level: str = "stable"             # stable | warning | drift


class DDMDetector:
    """
    Drift Detection Method для бинарного потока correct/incorrect.

    Формула (раздел 2.2 ВКР):
      p_t = n_err / n_t
      s_t = sqrt( p_t (1 - p_t) / n_t )
      warning: p_t + s_t ≥ p_min + 2·s_min
      drift:   p_t + s_t ≥ p_min + 3·s_min
    """

    def __init__(
        self,
        warning_level: float = DDM_WARNING_LEVEL,
        drift_level: float = DDM_DRIFT_LEVEL,
        min_samples: int = 30,
    ):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_samples = min_samples
        self.state = DDMState()

    def reset(self) -> None:
        self.state = DDMState()

    def add(self, correct: int) -> str:
        """
        Добавляет одно наблюдение (0 если ошибка, 1 если правильно).
        Возвращает уровень: stable | warning | drift.
        """
        s = self.state
        s.n += 1
        if correct == 0:
            s.n_errors += 1

        if s.n < self.min_samples:
            s.level = "stable"
            return s.level

        p = s.n_errors / s.n
        std_err = math.sqrt(max(p * (1 - p) / s.n, 1e-12))

        s.last_p = p
        s.last_s = std_err

        if p + std_err < s.p_min + s.s_min:
            s.p_min = p
            s.s_min = std_err

        if p + std_err >= s.p_min + self.drift_level * s.s_min:
            s.level = "drift"
        elif p + std_err >= s.p_min + self.warning_level * s.s_min:
            s.level = "warning"
        else:
            s.level = "stable"
        return s.level

    def fit_batch(self, correctness: List[int]) -> str:
        """Применить к батчу. Возвращает итоговый уровень."""
        self.reset()
        level = "stable"
        for c in correctness:
            level = self.add(c)
        return level

    @property
    def error_rate(self) -> float:
        return self.state.last_p
