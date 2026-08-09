"""
Единая точка обучения LSTM + CatBoost Ensemble.
Используется в:
cold_start.py (первичное обучение)
meta_agent.py (переобучение по решению LinUCB)
Интерфейс:
TrainConfig — параметры обучения (передаются из мета-агента)
train_all(cfg) — обучает обе модели
has_active_models() — проверка перед запуском
get_active_models_info() — мета-данные активных моделей (для LinUCB reward)
"""
from __future__ import annotations
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
from app.config import (
    DATA_DIR, FINAL_DATASET_NAME,
    LSTM_SEQ_LEN, LSTM_HORIZON, LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LEARNING_RATE,
)
from app.db import get_conn
from app.ml.catboost_model import (
    train_catboost_ensemble, N_ENSEMBLE, SYNTHETIC_WEIGHT,
)
from app.ml.lstm_model import train_lstm

logger = logging.getLogger(__name__)

# ============================================================
# КОНФИГУРАЦИЯ ОБУЧЕНИЯ
# ============================================================
@dataclass
class TrainConfig:
    """
    Параметры обучения, выбираемые Головой 2 LinUCB бандита.
    Каждый из 8 пресетов в meta_agent.py создаёт TrainConfig с разными значениями.
    """
    # Глобальные
    history_window_days: Optional[int] = None  # None = вся история

    # LSTM
    lookback: int = LSTM_SEQ_LEN
    lstm_learning_rate: float = LSTM_LEARNING_RATE
    lstm_epochs: int = LSTM_EPOCHS
    lstm_units: Tuple[int, int] = (64, 32)      # размеры LSTM-слоёв
    lstm_dense_units: int = 16                   # размер Dense перед выходом

    # CatBoost ensemble
    class_weights_mode: str = "balanced"  # "balanced" | "boost_jump" | "inverse"
    n_ensemble: int = N_ENSEMBLE          # размер ансамбля
    synthetic_weight: float = SYNTHETIC_WEIGHT
    catboost_learning_rate: float = 0.03  # <-- настраиваемый LR
    catboost_test_only_real: bool = False # <-- True = тест только на REAL_DIRECTIONS

    def to_dict(self) -> dict:
        d = asdict(self)
        # Tuple не сериализуется в JSON — преобразуем в list
        d["lstm_units"] = list(d["lstm_units"])
        return d

# ============================================================
# ЗАГРУЗКА ОБУЧАЮЩИХ ДАННЫХ
# ============================================================
def _load_training_dataset(history_window_days: Optional[int] = None) -> pd.DataFrame:
    """
    Загружает обучающий датасет.
    Приоритет:
      1. Если есть data/final_data_set_train.csv — используем его (preprocessed).
      2. Иначе берём из таблицы observations.

    Это позволяет:
      - cold_start: использовать пред готовленный CSV для быстрого старта
      - meta_agent retrain: использовать актуальные данные из observations
    """
    csv_path = DATA_DIR / FINAL_DATASET_NAME
    if csv_path.exists():
        logger.info(f"Loading from prepared CSV: {csv_path}")
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    else:
        logger.info("Loading from observations table")
        with get_conn() as c:
            df = pd.read_sql_query(
                "SELECT * FROM observations ORDER BY option_name, observation_date, time_encoded",
                c,
            )

    # Нормализация полей (поддерживаем оба формата CSV и SQL)
    if "observation_date" not in df.columns and "date" in df.columns:
        df["observation_date"] = df["date"]
    df["observation_date"] = pd.to_datetime(df["observation_date"])

    if "option_name" not in df.columns:
        df["option_name"] = df["direction"].astype(str) + "_" + df["option"].astype(str)
    if "time_slot" not in df.columns and "time" in df.columns:
        df["time_slot"] = df["time"]
    if "profile" not in df.columns:
        df["profile"] = "unknown"

    # Фильтр окна
    if history_window_days is not None:
        cutoff = df["observation_date"].max() - pd.Timedelta(days=history_window_days)
        df = df[df["observation_date"] >= cutoff].reset_index(drop=True)
        logger.info(f"After history window {history_window_days}d: {len(df)} rows")

    return df

# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ
# ============================================================
def train_all(cfg: Optional[TrainConfig] = None) -> Dict:
    """
    Обучает обе модели.
    Возвращает dict совместимый с cold_start._log_training_metrics:
      {
         "duration_seconds": float,
         "lstm":     {"version", "metrics": {test, baseline_last_price_mae, ...}},
         "catboost": {"version", "metrics": {horizon, catboost: {test}, ...}},
      }
    """
    cfg = cfg or TrainConfig()
    logger.info("=" * 60)
    logger.info("train_all started")
    logger.info(f"  config: {cfg.to_dict()}")
    logger.info("=" * 60)

    t0 = time.time()
    df = _load_training_dataset(history_window_days=cfg.history_window_days)
    logger.info(f"Dataset: {len(df)} rows, {df['option_name'].nunique()} options")

    # 1) LSTM v2.1
    logger.info("--- Training LSTM v2.1 ---")
    lstm_result = train_lstm(
        df=df,
        seq_len=cfg.lookback,
        horizon=LSTM_HORIZON,
        epochs=cfg.lstm_epochs,
        batch_size=LSTM_BATCH_SIZE,
        learning_rate=cfg.lstm_learning_rate,
        lstm_units=cfg.lstm_units,
        dense_units=cfg.lstm_dense_units,
    )

    # Адаптируем структуру метрик LSTM под ожидания cold_start
    lstm_metrics = lstm_result["metrics"]
    lstm_metrics_adapted = {
        # для cold_start._log_training_metrics: lstm.metrics.test.mae
        "test": {
            "mae": lstm_metrics["test"]["mae_rub"],
            "rmse": lstm_metrics["test"]["rmse_rub"],
            "r2": lstm_metrics["test"]["r2"],
            "direction_accuracy": lstm_metrics["test"]["direction_accuracy"],
        },
        "baseline_last_price_mae": lstm_metrics["baseline_naive"]["mae_rub"],
        "baseline_last_price_rmse": lstm_metrics["baseline_naive"]["rmse_rub"],
        "improvement_vs_baseline_rub": lstm_metrics["improvement_vs_baseline_rub"],
        "improvement_rmse_vs_baseline_rub": lstm_metrics["improvement_rmse_vs_baseline_rub"],
    }

    # 2) CatBoost Ensemble (15 моделей)
    logger.info("--- Training CatBoost Ensemble ---")
    cb_result = train_catboost_ensemble(
        df=df,
        n_ensemble=cfg.n_ensemble,
        class_weights_mode=cfg.class_weights_mode,
        synthetic_weight=cfg.synthetic_weight,
        learning_rate=cfg.catboost_learning_rate,
        test_only_real=cfg.catboost_test_only_real,
    )

    duration = time.time() - t0
    logger.info(f"train_all finished in {duration:.1f}s")

    return {
        "duration_seconds": duration,
        "config": cfg.to_dict(),
        "lstm": {
            "version": lstm_result["version"],
            "metrics": lstm_metrics_adapted,
        },
        "catboost": {
            "version": cb_result["version"],
            "metrics": cb_result["metrics"],   # уже в нужной структуре
        },
    }

# ============================================================
# ИНФОРМАЦИЯ ОБ АКТИВНЫХ МОДЕЛЯХ
# ============================================================
def has_active_models() -> bool:
    """True если активны обе модели (lstm и catboost)."""
    with get_conn() as c:
        rows = c.execute(
            "SELECT model_name FROM models_meta WHERE is_active=1"
        ).fetchall()
        active_names = {r["model_name"] for r in rows}
        return {"lstm", "catboost"}.issubset(active_names)

def get_active_models_info() -> Dict[str, Dict]:
    """
    Возвращает мета-данные активных моделей.
    Используется meta_agent.py для расчёта reward LinUCB:
      lstm_mae_before  ← info["lstm"]["metric_value"]
      catboost_f1_before ← info["catboost"]["metric_value"]   ← F1 АНСАМБЛЯ

    Поскольку catboost регистрируется как ОДНА запись с метрикой = F1 ансамбля,
    LinUCB reward работает корректно: ΔF1 = F1_ensemble_after - F1_ensemble_before.
    """
    info: Dict[str, Dict] = {}
    with get_conn() as c:
        rows = c.execute(
            """
            SELECT model_name, version, metric_name, metric_value,
                   model_path, meta_path, trained_at
            FROM models_meta
            WHERE is_active=1
            """
        ).fetchall()
        for r in rows:
            info[r["model_name"]] = {
                "version": r["version"],
                "metric_name": r["metric_name"],
                "metric_value": float(r["metric_value"]),
                "model_path": r["model_path"],
                "meta_path": r["meta_path"],
                "trained_at": r["trained_at"],
            }
    return info