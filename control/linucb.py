"""
LinUCB — контекстный многорукий бандит с линейной моделью награды.

Формулы из раздела 1.4 главы 1 ВКР:
  E[r | x, a] = xᵀθ_a
  θ̂_a = A_a⁻¹ · b_a
  UCB_a(x) = xᵀθ̂_a + α · sqrt(xᵀ A_a⁻¹ x)

Действие выбирается как argmax по UCB. После наблюдения реальной награды r:
  A_a ← A_a + x xᵀ
  b_a ← b_a + r · x

Состояние (A_a, b_a для каждой руки) сериализуется в SQLite таблицу bandit_state
через numpy .save() / .load() в блобы, что позволяет переживать рестарты.
"""
from __future__ import annotations

import io
import logging
from typing import Dict, List, Optional

import numpy as np

from app.config import LINUCB_ALPHA
from app.db import get_conn

logger = logging.getLogger(__name__)


def _numpy_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _blob_to_numpy(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


class LinUCBBandit:
    """
    Контекстный бандит с линейной моделью.

    Параметры:
      head_id   — идентификатор "головы" в таблице bandit_state ('head1' | 'head2')
      arm_ids   — список доступных действий (arm_id → str)
      context_dim — размерность вектора контекста
      alpha     — коэффициент исследования (по умолчанию из config)
    """

    def __init__(
        self,
        head_id: str,
        arm_ids: List[str],
        context_dim: int,
        alpha: float = LINUCB_ALPHA,
    ):
        if not arm_ids:
            raise ValueError("arm_ids must be non-empty")
        self.head_id = head_id
        self.arm_ids = list(arm_ids)
        self.context_dim = context_dim
        self.alpha = alpha

        # A_a ∈ R^{d×d}, b_a ∈ R^d — инициализация единичной матрицей
        self.A: Dict[str, np.ndarray] = {}
        self.b: Dict[str, np.ndarray] = {}
        self.n_pulls: Dict[str, int] = {arm: 0 for arm in arm_ids}
        self.cumulative_reward: Dict[str, float] = {arm: 0.0 for arm in arm_ids}

        self._load_or_init()

    # ============================================================
    # ПЕРСИСТЕНТНОСТЬ
    # ============================================================
    def _load_or_init(self) -> None:
        """Загружает состояние из bandit_state или инициализирует с нуля."""
        with get_conn() as c:
            rows = c.execute(
                "SELECT arm_id, a_matrix_blob, b_vector_blob, n_pulls, cumulative_reward "
                "FROM bandit_state WHERE head = ?",
                (self.head_id,),
            ).fetchall()

        existing = {r["arm_id"]: r for r in rows}

        for arm in self.arm_ids:
            if arm in existing:
                r = existing[arm]
                try:
                    A = _blob_to_numpy(r["a_matrix_blob"])
                    b = _blob_to_numpy(r["b_vector_blob"])
                    if A.shape == (self.context_dim, self.context_dim) and b.shape == (self.context_dim,):
                        self.A[arm] = A
                        self.b[arm] = b
                        self.n_pulls[arm] = int(r["n_pulls"])
                        self.cumulative_reward[arm] = float(r["cumulative_reward"])
                        continue
                    else:
                        logger.warning(
                            f"[{self.head_id}/{arm}] shape mismatch "
                            f"({A.shape}, {b.shape}) vs expected "
                            f"({self.context_dim}x{self.context_dim}). Reinit."
                        )
                except Exception as e:
                    logger.warning(f"Failed to load {self.head_id}/{arm}: {e}. Reinit.")

            # Инициализация с нуля
            self.A[arm] = np.eye(self.context_dim, dtype=np.float64)
            self.b[arm] = np.zeros(self.context_dim, dtype=np.float64)

        self._save_all()

    def _save_arm(self, arm: str) -> None:
        a_blob = _numpy_to_blob(self.A[arm])
        b_blob = _numpy_to_blob(self.b[arm])
        with get_conn() as c:
            c.execute(
                """
                INSERT INTO bandit_state (head, arm_id, a_matrix_blob, b_vector_blob, n_pulls, cumulative_reward, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(head, arm_id) DO UPDATE SET
                    a_matrix_blob = excluded.a_matrix_blob,
                    b_vector_blob = excluded.b_vector_blob,
                    n_pulls = excluded.n_pulls,
                    cumulative_reward = excluded.cumulative_reward,
                    updated_at = excluded.updated_at
                """,
                (self.head_id, arm, a_blob, b_blob,
                 self.n_pulls[arm], self.cumulative_reward[arm]),
            )
            c.commit()

    def _save_all(self) -> None:
        for arm in self.arm_ids:
            self._save_arm(arm)

    def reset(self) -> None:
        """Сбросить всё состояние (для главы 2.4: при retrain статистика сбрасывается)."""
        for arm in self.arm_ids:
            self.A[arm] = np.eye(self.context_dim, dtype=np.float64)
            self.b[arm] = np.zeros(self.context_dim, dtype=np.float64)
            self.n_pulls[arm] = 0
            self.cumulative_reward[arm] = 0.0
        self._save_all()

    # ============================================================
    # LINUCB ЛОГИКА
    # ============================================================
    def _ucb_score(self, arm: str, x: np.ndarray) -> float:
        """Верхняя доверительная граница для действия arm на контексте x."""
        A_inv = np.linalg.inv(self.A[arm])
        theta = A_inv @ self.b[arm]
        mean_reward = float(x @ theta)
        bonus = self.alpha * float(np.sqrt(max(x @ A_inv @ x, 0.0)))
        return mean_reward + bonus

    def select_arm(self, context: np.ndarray) -> Dict:
        """
        Выбирает действие с максимальной UCB-оценкой.
        Возвращает словарь: chosen_arm, scores, mean_rewards, bonuses.
        """
        x = self._validate_context(context)
        scores, means, bonuses = {}, {}, {}
        for arm in self.arm_ids:
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            mean_r = float(x @ theta)
            bonus = self.alpha * float(np.sqrt(max(x @ A_inv @ x, 0.0)))
            means[arm] = mean_r
            bonuses[arm] = bonus
            scores[arm] = mean_r + bonus

        chosen = max(scores, key=scores.get)
        return {
            "chosen_arm": chosen,
            "scores": scores,
            "mean_rewards": means,
            "bonuses": bonuses,
        }

    def update(self, arm: str, context: np.ndarray, reward: float) -> None:
        """
        Обновляет A и b для выбранной руки после наблюдения награды.
            A_a ← A_a + x xᵀ
            b_a ← b_a + r · x
        """
        if arm not in self.arm_ids:
            raise ValueError(f"Unknown arm: {arm}")
        x = self._validate_context(context)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.n_pulls[arm] += 1
        self.cumulative_reward[arm] += reward
        self._save_arm(arm)

    def _validate_context(self, context: np.ndarray) -> np.ndarray:
        x = np.asarray(context, dtype=np.float64).flatten()
        if x.shape != (self.context_dim,):
            raise ValueError(
                f"context shape mismatch: got {x.shape}, expected ({self.context_dim},)"
            )
        return x

    # ============================================================
    # ДИАГНОСТИКА
    # ============================================================
    def summary(self) -> Dict:
        """Сводка по рукам — число вытягиваний, средние награды."""
        return {
            arm: {
                "n_pulls": self.n_pulls[arm],
                "cumulative_reward": self.cumulative_reward[arm],
                "avg_reward": (
                    self.cumulative_reward[arm] / self.n_pulls[arm]
                    if self.n_pulls[arm] > 0 else 0.0
                ),
            }
            for arm in self.arm_ids
        }
