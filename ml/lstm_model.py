"""
LSTM модель для регрессионного прогноза цены (v2.1).

Ключевые изменения:
- Используем готовые фичи из БД/CSV (price_lag, EMA, RSI и т.д.)
- best_price включена как фича (не leakage — текущая цена известна)
- Архитектура настраиваемая (по умолчанию 64/32/16)
- Взвешенный Huber loss: штрафует сильнее за ошибки на скачках
- Таргет: ratio future/current (для совместимости с взвешенным loss)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from app.config import (
    LSTM_SEQ_LEN, LSTM_HORIZON, LSTM_EPOCHS, LSTM_BATCH_SIZE,
    LSTM_LEARNING_RATE, LSTM_RANDOM_SEED, MODELS_DIR,
)
from app.db import get_conn, register_model_version, set_active_model

logger = logging.getLogger(__name__)

LSTM_MODEL_DIR = MODELS_DIR / "lstm"
LSTM_MODEL_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_FACTOR = 10.0

# Готовые фичи, которые ожидаем найти в БД (построены feature_builder.py)
BASE_FEATURE_COLS = [
    "best_price",
    "price_lag_1", "price_lag_2", "price_lag_3",
    "rolling_mean_3d", "rolling_std_3d",
    "ema_3_day", "ema_7_day",
    "price_momentum", "price_acceleration",
    "prev_price_change", "rsi_7",
    "day_of_week_sin", "day_of_week_cos",
    "days_until_departure", "time_encoded", "flight_duration_minutes", "usd_rate",
]


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Проверяет наличие всех BASE_FEATURE_COLS.
    Если фичей нет (сырые данные), достраивает базовый набор из best_price.
    """
    df = df.copy()

    # Если нет готовых фичей, но есть best_price — считаем базовый набор
    if "price_lag_1" not in df.columns and "best_price" in df.columns:
        df = df.sort_values(["option_name", "observation_date", "time_encoded"]).reset_index(drop=True)
        out = []
        for _, g in df.groupby("option_name", sort=False):
            g = g.copy().reset_index(drop=True)
            p = g["best_price"].astype(float)

            for lag in [1, 2, 3]:
                g[f"price_lag_{lag}"] = p.shift(lag)

            g["rolling_mean_3d"] = p.rolling(3, min_periods=1).mean()
            g["rolling_std_3d"] = p.rolling(3, min_periods=1).std().fillna(0)
            g["ema_3_day"] = p.ewm(span=3, adjust=False).mean()
            g["ema_7_day"] = p.ewm(span=7, adjust=False).mean()

            momentum = p.pct_change().fillna(0) * 100
            g["price_momentum"] = momentum
            g["price_acceleration"] = momentum.diff().fillna(0)
            g["prev_price_change"] = momentum.shift(1).fillna(0)

            delta = p.diff()
            gain = delta.clip(lower=0).rolling(7, min_periods=1).mean()
            loss = (-delta.clip(upper=0)).rolling(7, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            g["rsi_7"] = (100 - 100 / (1 + rs)).fillna(50)

            dow = g["day_of_week"].astype(float)
            g["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
            g["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)

            out.append(g)
        df = pd.concat(out, ignore_index=True)

    # Заполняем отсутствующие нулями / средними
    for c in BASE_FEATURE_COLS:
        if c not in df.columns:
            logger.warning(f"Feature {c} missing, filling with 0")
            df[c] = 0.0

    return df


def _build_target_ratio(df: pd.DataFrame, horizon: int = LSTM_HORIZON) -> pd.DataFrame:
    """Ratio future/current через horizon шагов (как в lstm_modeling.py)."""
    parts = []
    for opt, g in df.groupby("option_name", sort=False):
        g = g.copy().reset_index(drop=True)
        bp = g["best_price"].values.astype(np.float32)
        future = np.full_like(bp, np.nan, dtype=np.float32)
        for i in range(len(bp) - horizon):
            future[i] = bp[i + horizon]
        g["_future_price"] = future
        g["target_ratio"] = (future / np.where(bp == 0, 1, bp)).astype(np.float32)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Собирает X, y из готовых фичей."""
    df = _ensure_features(df)

    dummies = pd.get_dummies(df["direction"], prefix="dir", dtype=np.float32)

    feat_df = pd.concat([df[BASE_FEATURE_COLS], dummies], axis=1).astype(np.float32)
    feat_df = feat_df.fillna(0)
    feature_names = list(feat_df.columns)

    X = feat_df.values.astype(np.float32)
    y = df["target_ratio"].values.astype(np.float32)
    bp = df["best_price"].values.astype(np.float32)
    fp = df["_future_price"].values.astype(np.float32)
    opts = df["option_name"].values
    return feat_df, y, bp, fp, opts, feature_names


def _build_sequences(X_df: pd.DataFrame, y: np.ndarray, bp: np.ndarray,
                     fp: np.ndarray, opts: np.ndarray, opt_filter: set,
                     seq_len: int):
    Xs, ys, bps, fps = [], [], [], []
    X = X_df.values
    for opt in opt_filter:
        idx = np.where(opts == opt)[0]
        if len(idx) < seq_len + 1:
            continue
        ox, oy = X[idx], y[idx]
        obp, ofp = bp[idx], fp[idx]
        for i in range(seq_len, len(ox)):
            Xs.append(ox[i - seq_len:i])
            ys.append(oy[i])
            bps.append(obp[i])
            fps.append(ofp[i])
    if not Xs:
        raise RuntimeError("No sequences built")
    return (np.array(Xs, dtype=np.float32),
            np.array(ys, dtype=np.float32),
            np.array(bps, dtype=np.float32),
            np.array(fps, dtype=np.float32))


def train_lstm(
        df: pd.DataFrame,
        seq_len: int = LSTM_SEQ_LEN,
        horizon: int = LSTM_HORIZON,
        epochs: int = LSTM_EPOCHS,
        batch_size: int = LSTM_BATCH_SIZE,
        learning_rate: float = LSTM_LEARNING_RATE,
        seed: int = LSTM_RANDOM_SEED,
        lstm_units: Tuple[int, int] = (64, 32),
        dense_units: int = 16,
) -> Dict:
    """Обучение LSTM v2.1."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

    logger.info(f"LSTM v2.1 training: {len(df)} rows, {df['option_name'].nunique()} options")
    logger.info(f"Architecture: LSTM{lstm_units} -> Dense({dense_units}), lr={learning_rate}")

    df = _build_target_ratio(df, horizon=horizon)
    df = df.dropna(subset=["target_ratio"]).reset_index(drop=True)
    logger.info(f"After dropna: {len(df)} rows")

    feat_df, y_arr, bp_arr, fp_arr, opts_arr, feature_names = _prepare_xy(df)

    # Split по опциям 70/15/15
    unique_opts = sorted(df["option_name"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_opts)
    n = len(unique_opts)
    train_opts = set(unique_opts[:int(n * 0.7)])
    val_opts = set(unique_opts[int(n * 0.7):int(n * 0.85)])
    test_opts = set(unique_opts[int(n * 0.85):])
    logger.info(f"Split: train={len(train_opts)}, val={len(val_opts)}, test={len(test_opts)}")

    X_tr, y_tr, bp_tr, fp_tr = _build_sequences(feat_df, y_arr, bp_arr, fp_arr, opts_arr, train_opts, seq_len)
    X_va, y_va, bp_va, fp_va = _build_sequences(feat_df, y_arr, bp_arr, fp_arr, opts_arr, val_opts, seq_len)
    X_te, y_te, bp_te, fp_te = _build_sequences(feat_df, y_arr, bp_arr, fp_arr, opts_arr, test_opts, seq_len)
    logger.info(f"Sequences: tr={len(X_tr)}, va={len(X_va)}, te={len(X_te)}")

    # Scaling
    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(-1, X_tr.shape[-1]))
    def scale(X):
        return scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    X_tr_s, X_va_s, X_te_s = scale(X_tr), scale(X_va), scale(X_te)

    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std()) if y_tr.std() > 1e-8 else 1.0
    y_tr_s = (y_tr - y_mean) / y_std
    y_va_s = (y_va - y_mean) / y_std

    # Взвешенный Huber loss (как в lstm_modeling.py)
    def weighted_huber(y_true, y_pred):
        weights = 1.0 + WEIGHT_FACTOR * tf.abs(y_true - 1.0)
        return tf.reduce_mean(weights * tf.keras.losses.huber(y_true, y_pred))

    # Модель
    model = Sequential([
        Input(shape=(seq_len, X_tr.shape[-1])),
        LSTM(lstm_units[0], return_sequences=True, recurrent_dropout=0.1),
        Dropout(0.2),
        LSTM(lstm_units[1], return_sequences=False, recurrent_dropout=0.1),
        Dropout(0.2),
        Dense(dense_units, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=weighted_huber, metrics=["mae"])

    early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_tr_s, y_tr_s,
              validation_data=(X_va_s, y_va_s),
              epochs=epochs, batch_size=batch_size,
              callbacks=[early], verbose=0)

    # Test metrics
    pred_s = model.predict(X_te_s, verbose=0).flatten()
    pred_ratio = pred_s * y_std + y_mean
    pred_price = bp_te * pred_ratio
    mae = float(mean_absolute_error(fp_te, pred_price))
    rmse = float(np.sqrt(mean_squared_error(fp_te, pred_price)))
    r2 = float(r2_score(fp_te, pred_price))
    naive_mae = float(mean_absolute_error(fp_te, bp_te))
    naive_rmse = float(np.sqrt(mean_squared_error(fp_te, bp_te)))
    direction_acc = float((np.sign(pred_ratio - 1.0) == np.sign(y_te - 1.0)).mean())

    logger.info(f"Test MAE={mae:.2f} RUB, RMSE={rmse:.2f}, R²={r2:.4f}")
    logger.info(f"Naive MAE={naive_mae:.2f}, RMSE={naive_rmse:.2f}")
    logger.info(f"Improvement RMSE: {naive_rmse - rmse:+.2f} RUB ({100*(naive_rmse-rmse)/naive_rmse:+.1f}%)")
    logger.info(f"Direction acc: {direction_acc:.3f}")

    # Save
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = LSTM_MODEL_DIR / f"model_{version}.keras"
    meta_path = LSTM_MODEL_DIR / f"meta_{version}.json"
    model.save(str(model_path))

    metrics_dict = {
        "test": {
            "mae_rub": mae, "rmse_rub": rmse, "r2": r2,
            "direction_accuracy": direction_acc,
        },
        "baseline_naive": {"mae_rub": naive_mae, "rmse_rub": naive_rmse},
        "improvement_vs_baseline_rub": naive_mae - mae,
        "improvement_rmse_vs_baseline_rub": naive_rmse - rmse,
    }
    meta = {
        "version": version,
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "y_mean": y_mean,
        "y_std": y_std,
        "config": {
            "seq_len": seq_len,
            "horizon": horizon,
            "target": "ratio",
            "epochs": epochs,
            "lstm_units": lstm_units,
            "dense_units": dense_units,
            "learning_rate": learning_rate,
        },
        "metrics": metrics_dict,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    register_model_version(
        model_name="lstm",
        version=version,
        model_path=str(model_path),
        meta_path=str(meta_path),
        metric_name="test_mae_rub",
        metric_value=mae,
    )
    set_active_model("lstm", version)

    logger.info(f"LSTM saved: {model_path}")
    return {"version": version, "metrics": metrics_dict, "model_path": str(model_path)}


def _load_lstm_artifacts(version: Optional[str] = None) -> Tuple[object, Dict]:
    if version is None:
        with get_conn() as c:
            row = c.execute(
                "SELECT version, model_path, meta_path FROM models_meta "
                "WHERE model_name='lstm' AND is_active=1 LIMIT 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("No active LSTM model")
        version = row["version"]
        model_path = row["model_path"]
        meta_path = row["meta_path"]
    else:
        model_path = str(LSTM_MODEL_DIR / f"model_{version}.keras")
        meta_path = str(LSTM_MODEL_DIR / f"meta_{version}.json")

    model = load_model(model_path, compile=False)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def predict_lstm(option_name: str) -> Optional[Dict]:
    try:
        model, meta = _load_lstm_artifacts()
    except Exception as e:
        logger.warning(f"LSTM load failed: {e}")
        return None

    seq_len = int(meta["config"]["seq_len"])
    feature_names = meta["feature_names"]
    scaler_mean = np.array(meta["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(meta["scaler_scale"], dtype=np.float32)
    y_mean = float(meta["y_mean"])
    y_std = float(meta["y_std"])

    # Запрашиваем все колонки из БД (готовые фичи + базовые)
    with get_conn() as c:
        df = pd.read_sql_query(
            "SELECT * FROM observations WHERE option_name = ? ORDER BY observation_date, time_encoded",
            c, params=(option_name,)
        )
    if len(df) < seq_len + 1:
        logger.warning(f"Not enough history for {option_name}: {len(df)} < {seq_len + 1}")
        return None

    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = _ensure_features(df)

    dummies = pd.get_dummies(df["direction"], prefix="dir", dtype=np.float32)
    feat_df = pd.concat([df[BASE_FEATURE_COLS], dummies], axis=1).astype(np.float32)

    for c_ in feature_names:
        if c_ not in feat_df.columns:
            feat_df[c_] = 0.0
    feat_df = feat_df[feature_names].fillna(0)

    seq_X = feat_df.tail(seq_len).values.astype(np.float32)
    if seq_X.shape != (seq_len, len(feature_names)):
        logger.warning(f"Bad seq shape: {seq_X.shape}, expected ({seq_len}, {len(feature_names)})")
        return None

    flat = (seq_X - scaler_mean) / np.where(scaler_scale == 0, 1, scaler_scale)
    flat = flat.reshape(1, seq_len, -1)

    pred_s = float(model.predict(flat, verbose=0).flatten()[0])
    pred_ratio = pred_s * y_std + y_mean
    current_price = float(df["best_price"].iloc[-1])
    pred_rub = current_price * pred_ratio

    return {
        "model_version": meta["version"],
        "current_price": current_price,
        "pred_ratio": float(pred_ratio),
        "pred_rub": float(pred_rub),
        "delta_pct": float((pred_ratio - 1.0) * 100),
        "delta_rub": float(pred_rub - current_price),   # ← ИСПРАВЛЕНО: абсолютное изменение в рублях
        "horizon": int(meta["config"]["horizon"]),
    }