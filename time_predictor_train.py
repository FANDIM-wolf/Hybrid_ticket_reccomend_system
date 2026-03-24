#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ФИНАЛЬНАЯ ВЕРСИЯ
LSTM для прогнозирования минимальной цены авиабилета
по временным рядам вида: direction + departure_date

Что делает скрипт:
1. Загружает lstm_timeseries_combined.csv
2. Очищает и нормализует данные
3. Для каждого маршрута агрегирует офферы в ряд:
   entity_id = direction + departure_date
   observation_ts -> min(price)
4. Делит каждый ряд по времени на train / val / test
5. Обучает scaler только на train
6. Строит окна для LSTM без смешивания разных вылетов
7. Подбирает seq_length по validation MAE
8. Выводит и сохраняет метрики train / val / test
"""

import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# =========================================================
# TensorFlow / Keras
# =========================================================
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam

    tf.random.set_seed(42)
    np.random.seed(42)
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# =========================================================
# Конфигурация
# =========================================================
HYPERPARAM_CONFIG = {
    "seq_lengths": [3, 5, 7],
    "epochs": 80,
    "batch_size": 32,
    "lstm_units": 32,
    "dropout": 0.20,
    "learning_rate": 0.001
}

ROUTES_CONFIG = {
    "MOW_OVB": {"direction": "MOW->OVB", "name": "Москва → Новосибирск"},
    "OVB_MOW": {"direction": "OVB->MOW", "name": "Новосибирск → Москва"},
    "MOW_SHA": {"direction": "MOW->SHA", "name": "Москва → Шанхай"},
    "MOW_NQZ": {"direction": "MOW->NQZ", "name": "Москва → Нур-Султан"},
    "MOW_KRR": {"direction": "MOW->KRR", "name": "Москва → Краснодар"},
    "KRR_MOW": {"direction": "KRR->MOW", "name": "Краснодар → Москва"},
    "MOW_TJM": {"direction": "MOW->TJM", "name": "Москва → Тюмень"},
    "TJM_MOW": {"direction": "TJM->MOW", "name": "Тюмень → Москва"},
    "MOW_LED": {"direction": "MOW->LED", "name": "Москва → Санкт-Петербург"},
    "LED_MOW": {"direction": "LED->MOW", "name": "Санкт-Петербург → Москва"},
    "MOW_KZN": {"direction": "MOW->KZN", "name": "Москва → Казань"},
}

MIN_POINTS_PER_SERIES = 8
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20


# =========================================================
# Утилиты
# =========================================================
def safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    return r2_score(y_true, y_pred)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def inverse_target(target_scaler, arr_1d):
    arr_2d = np.asarray(arr_1d).reshape(-1, 1)
    return target_scaler.inverse_transform(arr_2d).ravel()


# =========================================================
# Загрузка и подготовка данных
# =========================================================
def load_and_prepare_data(filepath="lstm_timeseries_combined.csv"):
    if not os.path.exists(filepath):
        print(f"❌ Файл не найден: {filepath}")
        return None

    df = pd.read_csv(filepath, low_memory=False)
    print(f"📊 Размер данных: {df.shape[0]} строк × {df.shape[1]} колонок")

    required_cols = [
        "observation_ts",
        "departure_at",
        "price",
        "airline",
        "origin",
        "destination"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ Отсутствуют обязательные колонки: {missing}")
        return None

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["observation_ts"] = pd.to_datetime(df["observation_ts"], errors="coerce", utc=True)
    df["departure_at"] = pd.to_datetime(df["departure_at"], errors="coerce", utc=True)

    df["origin"] = df["origin"].astype(str).str.upper().str.strip()
    df["destination"] = df["destination"].astype(str).str.upper().str.strip()
    df["airline"] = df["airline"].astype(str).fillna("UNK").replace("nan", "UNK")

    if "direction" not in df.columns or df["direction"].isna().all():
        df["direction"] = df["origin"] + "->" + df["destination"]
    else:
        df["direction"] = df["direction"].astype(str).str.upper().str.strip()

    if "days_before_departure" not in df.columns:
        delta = df["departure_at"] - df["observation_ts"]
        df["days_before_departure"] = delta.dt.total_seconds() / 86400.0
    else:
        df["days_before_departure"] = pd.to_numeric(df["days_before_departure"], errors="coerce")

    df = df.dropna(subset=[
        "price",
        "observation_ts",
        "departure_at",
        "direction",
        "days_before_departure"
    ])

    df = df.sort_values(["direction", "departure_at", "observation_ts"]).reset_index(drop=True)

    print(f"✅ После очистки: {len(df)} строк")
    print(f"📅 Период наблюдений: {df['observation_ts'].min()} — {df['observation_ts'].max()}")
    print(f"🛫 Уникальных направлений: {df['direction'].nunique()}")

    return df


# =========================================================
# Фильтрация маршрутов
# =========================================================
def filter_routes(df):
    filtered_data = {}

    for route_key, config in ROUTES_CONFIG.items():
        route_df = df[df["direction"] == config["direction"]].copy()
        if len(route_df) > 0:
            filtered_data[route_key] = {
                "data": route_df,
                "config": config
            }

    print(f"\n🔍 Найдено маршрутов: {len(filtered_data)}")
    for route_key, info in filtered_data.items():
        print(f"  • {info['config']['name']}: {len(info['data'])} строк")

    if not filtered_data:
        print("\n⚠️ Подходящие маршруты не найдены.")
        print("Доступные направления:")
        for direction, cnt in df["direction"].value_counts().head(20).items():
            print(f"  • {direction}: {cnt}")

    return filtered_data


# =========================================================
# Агрегация в временные ряды
# =========================================================
def aggregate_route_timeseries(route_df):
    """
    Создаем ряды:
    entity_id = direction + departure_date

    На каждом observation_ts берем:
    - минимальную цену
    - минимальные дни до вылета
    - число офферов
    - число авиакомпаний
    """
    df = route_df.copy()
    df["departure_date"] = df["departure_at"].dt.floor("D")

    agg = (
        df.groupby(["direction", "departure_date", "observation_ts"], as_index=False)
        .agg(
            price=("price", "min"),
            days_before_departure=("days_before_departure", "min"),
            offers_count=("price", "size"),
            airlines_count=("airline", pd.Series.nunique),
        )
    )

    agg["departure_weekday"] = agg["departure_date"].dt.dayofweek.astype(int)
    agg["departure_month"] = agg["departure_date"].dt.month.astype(int)
    agg["observation_weekday"] = agg["observation_ts"].dt.dayofweek.astype(int)

    agg["entity_id"] = (
            agg["direction"].astype(str)
            + "|"
            + agg["departure_date"].dt.strftime("%Y-%m-%d")
    )

    agg = agg.sort_values(["entity_id", "observation_ts"]).reset_index(drop=True)
    return agg


def get_feature_columns():
    return [
        "price",
        "days_before_departure",
        "offers_count",
        "airlines_count",
        "departure_weekday",
        "departure_month",
        "observation_weekday",
    ]


# =========================================================
# Разделение рядов
# =========================================================
def split_entity_indices(n):
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_end = max(train_end, 1)
    val_end = max(val_end, train_end + 1)

    if val_end >= n:
        val_end = n - 1

    return train_end, val_end


def collect_eligible_entities(agg_df, seq_length):
    eligible = []
    skipped = []

    for entity_id, g in agg_df.groupby("entity_id"):
        g = g.sort_values("observation_ts").reset_index(drop=True)
        n = len(g)

        if n < max(MIN_POINTS_PER_SERIES, seq_length + 3):
            skipped.append((entity_id, n, "too_short"))
            continue

        train_end, val_end = split_entity_indices(n)

        if train_end <= seq_length:
            skipped.append((entity_id, n, "train_too_short"))
            continue

        if (val_end - train_end) < 1:
            skipped.append((entity_id, n, "no_val"))
            continue

        if (n - val_end) < 1:
            skipped.append((entity_id, n, "no_test"))
            continue

        eligible.append(g)

    return eligible, skipped


# =========================================================
# Масштабирование без leakage
# =========================================================
def fit_scalers_on_train(entities, feature_cols):
    train_feature_blocks = []
    train_target_blocks = []

    for g in entities:
        n = len(g)
        train_end, _ = split_entity_indices(n)
        train_part = g.iloc[:train_end].copy()

        train_feature_blocks.append(train_part[feature_cols].astype(np.float32))
        train_target_blocks.append(train_part[["price"]].astype(np.float32))

    X_train_all = pd.concat(train_feature_blocks, axis=0)
    y_train_all = pd.concat(train_target_blocks, axis=0)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    feature_scaler.fit(X_train_all)
    target_scaler.fit(y_train_all)

    return feature_scaler, target_scaler


# =========================================================
# Создание окон
# =========================================================
def build_windows(entities, seq_length, feature_cols, feature_scaler, target_scaler):
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for g in entities:
        g = g.sort_values("observation_ts").reset_index(drop=True)
        n = len(g)
        train_end, val_end = split_entity_indices(n)

        features_scaled = feature_scaler.transform(g[feature_cols].astype(np.float32))
        target_scaled = target_scaler.transform(g[["price"]].astype(np.float32)).ravel()

        for i in range(seq_length, n):
            x_window = features_scaled[i - seq_length:i]
            y_target = target_scaled[i]

            if i < train_end:
                X_train.append(x_window)
                y_train.append(y_target)
            elif i < val_end:
                X_val.append(x_window)
                y_val.append(y_target)
            else:
                X_test.append(x_window)
                y_test.append(y_target)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    meta = {
        "train_windows": len(X_train),
        "val_windows": len(X_val),
        "test_windows": len(X_test),
        "n_features": len(feature_cols)
    }

    return X_train, y_train, X_val, y_val, X_test, y_test, meta


# =========================================================
# Модель
# =========================================================
def build_lstm_model(input_shape, lstm_units=32, dropout=0.2, learning_rate=0.001):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(lstm_units, return_sequences=True),
        Dropout(dropout),
        LSTM(max(lstm_units // 2, 16), return_sequences=False),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model


def train_lstm_model(
        X_train, y_train, X_val, y_val,
        epochs=80, batch_size=32,
        lstm_units=32, dropout=0.2, learning_rate=0.001
):
    if not TF_AVAILABLE:
        print("❌ TensorFlow не установлен.")
        return None

    if len(X_train) < 10 or len(X_val) < 5:
        print("⚠️ Недостаточно данных для обучения.")
        return None

    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        lstm_units=lstm_units,
        dropout=dropout,
        learning_rate=learning_rate
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=0
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks
    )

    return {
        "model": model,
        "history": history,
        "actual_epochs": len(history.history["loss"])
    }


# =========================================================
# Оценка
# =========================================================
def calc_metrics(y_true_scaled, y_pred_scaled, target_scaler):
    y_true = inverse_target(target_scaler, y_true_scaled)
    y_pred = inverse_target(target_scaler, y_pred_scaled)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(safe_r2(y_true, y_pred))
    }


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, target_scaler):
    train_pred = model.predict(X_train, verbose=0).ravel()
    val_pred = model.predict(X_val, verbose=0).ravel()
    test_pred = model.predict(X_test, verbose=0).ravel()

    metrics = {
        "train": calc_metrics(y_train, train_pred, target_scaler),
        "val": calc_metrics(y_val, val_pred, target_scaler),
        "test": calc_metrics(y_test, test_pred, target_scaler)
    }

    predictions = {
        "train_true": inverse_target(target_scaler, y_train),
        "train_pred": inverse_target(target_scaler, train_pred),
        "val_true": inverse_target(target_scaler, y_val),
        "val_pred": inverse_target(target_scaler, val_pred),
        "test_true": inverse_target(target_scaler, y_test),
        "test_pred": inverse_target(target_scaler, test_pred)
    }

    return metrics, predictions


# =========================================================
# Подбор по маршруту
# =========================================================
def tune_route_lstm(route_df, route_key, route_name, save_dir="models_lstm_clean"):
    ensure_dir(save_dir)

    agg_df = aggregate_route_timeseries(route_df)
    feature_cols = get_feature_columns()

    best_record = None
    all_records = []

    print("\n" + "=" * 80)
    print(f"🧠 Маршрут: {route_name}")
    print("=" * 80)

    print(f"   Агрегированных точек: {len(agg_df)}")
    print(f"   Уникальных рядов: {agg_df['entity_id'].nunique()}")

    for seq_length in HYPERPARAM_CONFIG["seq_lengths"]:
        print(f"\n📐 seq_length = {seq_length}")

        entities, skipped = collect_eligible_entities(agg_df, seq_length)

        if len(entities) == 0:
            print("   ⚠️ Нет пригодных рядов для обучения.")
            continue

        feature_scaler, target_scaler = fit_scalers_on_train(entities, feature_cols)

        X_train, y_train, X_val, y_val, X_test, y_test, windows_meta = build_windows(
            entities=entities,
            seq_length=seq_length,
            feature_cols=feature_cols,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler
        )

        if len(X_train) < 10 or len(X_val) < 5 or len(X_test) < 5:
            print("   ⚠️ Слишком мало окон после разбиения.")
            continue

        train_result = train_lstm_model(
            X_train, y_train, X_val, y_val,
            epochs=HYPERPARAM_CONFIG["epochs"],
            batch_size=HYPERPARAM_CONFIG["batch_size"],
            lstm_units=HYPERPARAM_CONFIG["lstm_units"],
            dropout=HYPERPARAM_CONFIG["dropout"],
            learning_rate=HYPERPARAM_CONFIG["learning_rate"]
        )

        if train_result is None:
            continue

        model = train_result["model"]
        history = train_result["history"]
        actual_epochs = train_result["actual_epochs"]

        metrics, predictions = evaluate_model(
            model, X_train, y_train, X_val, y_val, X_test, y_test, target_scaler
        )

        record = {
            "seq_length": seq_length,
            "model": model,
            "history": history.history,
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "metrics": metrics,
            "predictions": predictions,
            "meta": {
                "entities_used": len(entities),
                "entities_skipped": len(skipped),
                "feature_cols": feature_cols,
                "train_windows": windows_meta["train_windows"],
                "val_windows": windows_meta["val_windows"],
                "test_windows": windows_meta["test_windows"],
                "actual_epochs": actual_epochs
            }
        }

        all_records.append(record)

        print(f"   entities used:  {len(entities)}")
        print(f"   entities skip:  {len(skipped)}")
        print(f"   train windows:  {windows_meta['train_windows']}")
        print(f"   val windows:    {windows_meta['val_windows']}")
        print(f"   test windows:   {windows_meta['test_windows']}")
        print(f"   actual epochs:  {actual_epochs}")

        print(f"   Train MAE:      {metrics['train']['mae']:.2f} ₽")
        print(f"   Train RMSE:     {metrics['train']['rmse']:.2f} ₽")
        print(f"   Train R²:       {metrics['train']['r2']:.4f}")

        print(f"   Val MAE:        {metrics['val']['mae']:.2f} ₽")
        print(f"   Val RMSE:       {metrics['val']['rmse']:.2f} ₽")
        print(f"   Val R²:         {metrics['val']['r2']:.4f}")

        print(f"   Test MAE:       {metrics['test']['mae']:.2f} ₽")
        print(f"   Test RMSE:      {metrics['test']['rmse']:.2f} ₽")
        print(f"   Test R²:        {metrics['test']['r2']:.4f}")

        if best_record is None or metrics["val"]["mae"] < best_record["metrics"]["val"]["mae"]:
            best_record = record

    if best_record is None:
        print("⚠️ Не удалось обучить модель для маршрута.")
        return None, None

    comparison_df = pd.DataFrame([
        {
            "seq_length": r["seq_length"],

            "train_mae": r["metrics"]["train"]["mae"],
            "train_rmse": r["metrics"]["train"]["rmse"],
            "train_r2": r["metrics"]["train"]["r2"],

            "val_mae": r["metrics"]["val"]["mae"],
            "val_rmse": r["metrics"]["val"]["rmse"],
            "val_r2": r["metrics"]["val"]["r2"],

            "test_mae": r["metrics"]["test"]["mae"],
            "test_rmse": r["metrics"]["test"]["rmse"],
            "test_r2": r["metrics"]["test"]["r2"],

            "entities_used": r["meta"]["entities_used"],
            "entities_skipped": r["meta"]["entities_skipped"],
            "train_windows": r["meta"]["train_windows"],
            "val_windows": r["meta"]["val_windows"],
            "test_windows": r["meta"]["test_windows"],
            "actual_epochs": r["meta"]["actual_epochs"],
        }
        for r in all_records
    ])

    model_path = os.path.join(save_dir, f"{route_key}_lstm_best.keras")
    meta_path = os.path.join(save_dir, f"{route_key}_metadata.joblib")
    hist_path = os.path.join(save_dir, f"{route_key}_history.joblib")
    cmp_path = os.path.join(save_dir, f"{route_key}_comparison.csv")
    pred_path = os.path.join(save_dir, f"{route_key}_predictions.csv")

    best_record["model"].save(model_path)

    metadata = {
        "route_key": route_key,
        "route_name": route_name,
        "best_seq_length": best_record["seq_length"],
        "metrics": best_record["metrics"],
        "meta": best_record["meta"],
        "train_date": datetime.now().isoformat()
    }
    joblib.dump(metadata, meta_path)
    joblib.dump(best_record["history"], hist_path)
    comparison_df.to_csv(cmp_path, index=False)

    pred_df = pd.DataFrame({
        "split": (
                ["train"] * len(best_record["predictions"]["train_true"]) +
                ["val"] * len(best_record["predictions"]["val_true"]) +
                ["test"] * len(best_record["predictions"]["test_true"])
        ),
        "y_true": np.concatenate([
            best_record["predictions"]["train_true"],
            best_record["predictions"]["val_true"],
            best_record["predictions"]["test_true"]
        ]),
        "y_pred": np.concatenate([
            best_record["predictions"]["train_pred"],
            best_record["predictions"]["val_pred"],
            best_record["predictions"]["test_pred"]
        ])
    })
    pred_df.to_csv(pred_path, index=False)

    print("\n✅ Лучшая конфигурация:")
    print(f"   Seq Length: {best_record['seq_length']}")
    print(f"   Train -> MAE: {best_record['metrics']['train']['mae']:.2f} ₽, "
          f"RMSE: {best_record['metrics']['train']['rmse']:.2f} ₽, "
          f"R²: {best_record['metrics']['train']['r2']:.4f}")
    print(f"   Val   -> MAE: {best_record['metrics']['val']['mae']:.2f} ₽, "
          f"RMSE: {best_record['metrics']['val']['rmse']:.2f} ₽, "
          f"R²: {best_record['metrics']['val']['r2']:.4f}")
    print(f"   Test  -> MAE: {best_record['metrics']['test']['mae']:.2f} ₽, "
          f"RMSE: {best_record['metrics']['test']['rmse']:.2f} ₽, "
          f"R²: {best_record['metrics']['test']['r2']:.4f}")

    return best_record, comparison_df


# =========================================================
# Главная функция
# =========================================================
def main():
    print("=" * 80)
    print("LSTM ДЛЯ ПРОГНОЗИРОВАНИЯ АВИАЦЕН БЕЗ LEAKAGE")
    print("=" * 80)

    if not TF_AVAILABLE:
        print("❌ TensorFlow не установлен. Установите: pip install tensorflow")
        return None

    filepath = "lstm_timeseries_combined.csv"
    df = load_and_prepare_data(filepath)

    if df is None or df.empty:
        print("❌ Данные не загружены.")
        return None

    route_data = filter_routes(df)
    if not route_data:
        print("❌ Подходящие маршруты не найдены.")
        return None

    save_dir = "models_lstm_clean"
    ensure_dir(save_dir)

    all_results = {}
    summary_rows = []

    print("\n" + "=" * 80)
    print("🚀 ОБУЧЕНИЕ ПО МАРШРУТАМ")
    print("=" * 80)

    for route_key, info in route_data.items():
        best_record, comparison_df = tune_route_lstm(
            route_df=info["data"],
            route_key=route_key,
            route_name=info["config"]["name"],
            save_dir=save_dir
        )

        if best_record is None:
            continue

        all_results[route_key] = {
            "best_seq_length": best_record["seq_length"],
            "metrics": best_record["metrics"],
            "meta": best_record["meta"],
            "comparison": comparison_df
        }

        summary_rows.append({
            "Маршрут": info["config"]["name"],
            "Seq Length": best_record["seq_length"],

            "Train MAE (₽)": round(best_record["metrics"]["train"]["mae"], 2),
            "Train RMSE (₽)": round(best_record["metrics"]["train"]["rmse"], 2),
            "Train R²": round(best_record["metrics"]["train"]["r2"], 4),

            "Val MAE (₽)": round(best_record["metrics"]["val"]["mae"], 2),
            "Val RMSE (₽)": round(best_record["metrics"]["val"]["rmse"], 2),
            "Val R²": round(best_record["metrics"]["val"]["r2"], 4),

            "Test MAE (₽)": round(best_record["metrics"]["test"]["mae"], 2),
            "Test RMSE (₽)": round(best_record["metrics"]["test"]["rmse"], 2),
            "Test R²": round(best_record["metrics"]["test"]["r2"], 4),

            "Entities": best_record["meta"]["entities_used"],
            "Train windows": best_record["meta"]["train_windows"],
            "Val windows": best_record["meta"]["val_windows"],
            "Test windows": best_record["meta"]["test_windows"],
            "Epochs": best_record["meta"]["actual_epochs"],
        })

    print("\n" + "=" * 80)
    print("📊 ИТОГОВАЯ СВОДКА")
    print("=" * 80)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)
    else:
        print("⚠️ Ни один маршрут не удалось корректно обучить.")

    print("\n" + "=" * 80)
    print("📁 СОХРАНЕННЫЕ ФАЙЛЫ")
    print("=" * 80)
    for f in sorted(os.listdir(save_dir)):
        print(f"  📄 {save_dir}/{f}")

    print("\n" + "=" * 80)
    print("✅ АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    results = main()
