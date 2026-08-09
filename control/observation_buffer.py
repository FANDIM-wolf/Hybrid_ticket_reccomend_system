"""
Буфер наблюдений модуля управления.

Хранит кортежи (прогноз, факт, остаток) для двух базовых моделей
и предоставляет их детектору дрейфа. Физически данные лежат в таблицах
predictions и residuals — класс-обёртка над SQL-операциями.

Ключевые операции:
  record_prediction()  — записать прогноз в момент его создания
  record_actual()      — когда реальная цена становится известна, обновить
                         строку predictions и вычислить остаток
  get_residuals()      — получить последние N остатков LSTM для Page-Hinkley
  get_error_rate()     — получить последние N флагов корректности CatBoost для DDM
"""
from __future__ import annotations

import logging
from typing import List, Optional

from app.db import get_conn

logger = logging.getLogger(__name__)


class ObservationBuffer:
    """Обёртка над таблицами predictions и residuals."""

    def record_prediction(
        self,
        option_name: str,
        predicted_at: str,
        target_time: str,
        current_price: float,
        lstm_pred_ratio: Optional[float],
        lstm_pred_rub: Optional[float],
        catboost_class: Optional[int],
        catboost_proba: Optional[List[float]],
        model_version: Optional[str] = None,
    ) -> int:
        """Сохраняет прогноз, возвращает prediction_id."""
        proba = catboost_proba or [None, None, None]
        while len(proba) < 3:
            proba.append(None)

        with get_conn() as c:
            cur = c.execute(
                """
                INSERT INTO predictions (
                    option_name, predicted_at, target_time, current_price,
                    lstm_pred_ratio, lstm_pred_rub,
                    catboost_class, catboost_proba_0, catboost_proba_1, catboost_proba_2,
                    model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    option_name, predicted_at, target_time, current_price,
                    lstm_pred_ratio, lstm_pred_rub,
                    catboost_class, proba[0], proba[1], proba[2],
                    model_version,
                ),
            )
            c.commit()
            return cur.lastrowid

    def record_actual(
        self,
        prediction_id: int,
        actual_price: float,
        actual_class: Optional[int] = None,
    ) -> None:
        """
        Обновляет строку predictions актуальным значением цены и записывает
        остаток в residuals. Вызывается, когда наступает target_time.
        """
        with get_conn() as c:
            row = c.execute(
                "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
            ).fetchone()
            if not row:
                logger.warning(f"prediction_id={prediction_id} not found")
                return

            c.execute(
                "UPDATE predictions SET actual_price = ? WHERE id = ?",
                (actual_price, prediction_id),
            )

            # LSTM residual: факт - прогноз (в процентах от текущей цены)
            lstm_residual = None
            if row["lstm_pred_rub"] is not None and row["current_price"]:
                actual_ratio = actual_price / row["current_price"]
                pred_ratio = row["lstm_pred_rub"] / row["current_price"]
                lstm_residual = actual_ratio - pred_ratio

            # CatBoost correctness
            cb_correct = None
            if row["catboost_class"] is not None and actual_class is not None:
                cb_correct = 1 if int(row["catboost_class"]) == int(actual_class) else 0

            c.execute(
                """
                INSERT INTO residuals (
                    prediction_id, option_name, observation_date,
                    lstm_residual, catboost_correct
                ) VALUES (?, ?, date('now'), ?, ?)
                """,
                (prediction_id, row["option_name"], lstm_residual, cb_correct),
            )
            c.commit()

    def get_lstm_residuals(self, limit: int = 200) -> List[float]:
        """Последние N остатков LSTM для Page-Hinkley."""
        with get_conn() as c:
            rows = c.execute(
                "SELECT lstm_residual FROM residuals "
                "WHERE lstm_residual IS NOT NULL "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r["lstm_residual"] for r in rows][::-1]

    def get_catboost_correctness(self, limit: int = 200) -> List[int]:
        """Последние N флагов корректности CatBoost для DDM."""
        with get_conn() as c:
            rows = c.execute(
                "SELECT catboost_correct FROM residuals "
                "WHERE catboost_correct IS NOT NULL "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r["catboost_correct"] for r in rows][::-1]

    def count(self) -> int:
        with get_conn() as c:
            return int(c.execute("SELECT COUNT(*) AS n FROM residuals").fetchone()["n"])
