#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostRegressor, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost не установлен. Установите: pip install catboost")


# =========================================================
# CONFIG
# =========================================================
RANDOM_STATE = 42
MIN_ROUTE_ROWS = 50

ROUTES_CONFIG = {
    "MOW_TAS": {"origin": "MOW", "destination": "TAS", "name": "Москва → Ташкент"},
    "MOW_SHA": {"origin": "MOW", "destination": "SHA", "name": "Москва → Шанхай"},
    "MOW_NQZ": {"origin": "MOW", "destination": "NQZ", "name": "Москва → Нур-Султан"},
    "KZN_MOW": {"origin": "KZN", "destination": "MOW", "name": "Казань → Москва"},
    "MOW_KRR": {"origin": "MOW", "destination": "KRR", "name": "Москва → Краснодар"},
    "KRR_MOW": {"origin": "KRR", "destination": "MOW", "name": "Краснодар → Москва"},
    "MOW_LED": {"origin": "MOW", "destination": "LED", "name": "Москва → Санкт-Петербург"},
    "LED_MOW": {"origin": "LED", "destination": "MOW", "name": "Санкт-Петербург → Москва"},
    "MOW_TJM": {"origin": "MOW", "destination": "TJM", "name": "Москва → Тюмень"},
    "TJM_MOW": {"origin": "TJM", "destination": "MOW", "name": "Тюмень → Москва"},
}

RF_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [6, 10],
    "min_samples_leaf": [1, 3],
}

CAT_PARAM_GRID = {
    "depth": [4, 6],
    "learning_rate": [0.03, 0.05],
    "l2_leaf_reg": [5, 10],
}


# =========================================================
# HELPERS
# =========================================================
def safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    return r2_score(y_true, y_pred)


def calc_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(safe_r2(y_true, y_pred))
    }


def first_existing(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


# =========================================================
# 1. LOAD + CANONICAL COLUMNS
# =========================================================
def build_canonical_columns(df):
    df = df.copy()

    if "price" not in df.columns:
        raise ValueError("Колонка 'price' не найдена")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    obs_col = first_existing(df, ["observation_ts", "run_at_utc"])
    if obs_col is None:
        raise ValueError("Не найдена observation_ts/run_at_utc")

    df["observation_ts_clean"] = pd.to_datetime(df[obs_col], errors="coerce", utc=True)

    dep_series = None
    for col in ["departure_at", "depart_date"]:
        if col in df.columns:
            cur = pd.to_datetime(df[col], errors="coerce", utc=True)
            dep_series = cur if dep_series is None else dep_series.fillna(cur)
    df["depart_dt_clean"] = dep_series if dep_series is not None else pd.NaT

    df["origin_clean"] = df["origin"].astype(str).str.upper().str.strip() if "origin" in df.columns else np.nan
    df["destination_clean"] = df["destination"].astype(str).str.upper().str.strip() if "destination" in df.columns else np.nan

    df["origin_airport_clean"] = (
        df["origin_airport"].astype(str).str.upper().str.strip()
        if "origin_airport" in df.columns else "UNK"
    )
    df["destination_airport_clean"] = (
        df["destination_airport"].astype(str).str.upper().str.strip()
        if "destination_airport" in df.columns else "UNK"
    )

    if "flight_number" in df.columns:
        df["flight_number_clean"] = df["flight_number"].astype(str).fillna("0").replace("nan", "0")
    else:
        df["flight_number_clean"] = "0"

    if "airline" in df.columns:
        df["airline_clean"] = df["airline"].astype(str).fillna("UNKNOWN").replace("nan", "UNKNOWN")
    else:
        df["airline_clean"] = "UNKNOWN"

    df["direction"] = df["origin_clean"].astype(str) + "->" + df["destination_clean"].astype(str)

    return df


def load_and_prepare_data(filepath="combined_prices_for_dates_oneway.csv"):
    print(f"📂 Загрузка файла: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"✅ Загружено: {len(df)} строк × {len(df.columns)} колонок")

    df = build_canonical_columns(df)

    df = df.dropna(subset=["price", "observation_ts_clean"])
    df = df[df["origin_clean"].notna() & df["destination_clean"].notna()]
    df = df[df["origin_clean"] != "NAN"]
    df = df[df["destination_clean"] != "NAN"]
    df = df[df["direction"] != "nan->nan"]

    if "depart_dt_clean" in df.columns:
        delta = df["depart_dt_clean"] - df["observation_ts_clean"]
        df["days_before_departure"] = delta.dt.total_seconds() / 86400.0
    else:
        df["days_before_departure"] = np.nan

    df["days_before_departure"] = df["days_before_departure"].replace([np.inf, -np.inf], np.nan)
    df["days_before_departure"] = df["days_before_departure"].fillna(0)

    print(f"✅ После очистки: {len(df)} записей")
    if len(df) > 0:
        print(f"📅 Период: {df['observation_ts_clean'].min()} — {df['observation_ts_clean'].max()}")

    return df


# =========================================================
# 2. ROUTES
# =========================================================
def filter_routes(df):
    filtered_data = {}

    for route_key, config in ROUTES_CONFIG.items():
        mask = (
                (df["origin_clean"] == config["origin"]) &
                (df["destination_clean"] == config["destination"])
        )
        route_df = df.loc[mask].copy()

        if len(route_df) == 0:
            continue

        route_df = route_df.sort_values("observation_ts_clean").reset_index(drop=True)
        route_df["snapshot_number"] = np.arange(1, len(route_df) + 1)

        filtered_data[route_key] = {
            "data": route_df,
            "config": config
        }

    print(f"🔍 Найдено маршрутов: {len(filtered_data)}")
    for route_key, info in filtered_data.items():
        print(f"  • {info['config']['name']}: {len(info['data'])} наблюдений")

    return filtered_data


# =========================================================
# 3. STATS
# =========================================================
def analyze_route_statistics(route_data):
    results = {}

    for route_key, info in route_data.items():
        df = info["data"]
        config = info["config"]

        mean_price = df["price"].mean()
        std_price = df["price"].std()

        results[route_key] = {
            "route_name": config["name"],
            "total_observations": int(len(df)),
            "unique_flights": int(df["flight_number_clean"].nunique()),
            "unique_airlines": int(df["airline_clean"].nunique()),
            "price_min": float(df["price"].min()),
            "price_max": float(df["price"].max()),
            "price_mean": float(mean_price),
            "price_median": float(df["price"].median()),
            "price_std": float(std_price),
            "price_cv": float((std_price / mean_price * 100) if mean_price > 0 else 0),
        }

    return results


# =========================================================
# 4. FEATURES
# =========================================================
def prepare_features(df):
    df = df.copy()

    df["days_before_departure"] = df["days_before_departure"].fillna(0)

    if "depart_dt_clean" in df.columns:
        df["departure_day_of_week"] = df["depart_dt_clean"].dt.dayofweek.fillna(0).astype(int)
        df["departure_month"] = df["depart_dt_clean"].dt.month.fillna(0).astype(int)
        df["departure_hour"] = df["depart_dt_clean"].dt.hour.fillna(0).astype(int)
        df["departure_day"] = df["depart_dt_clean"].dt.day.fillna(0).astype(int)
    else:
        df["departure_day_of_week"] = 0
        df["departure_month"] = 0
        df["departure_hour"] = 0
        df["departure_day"] = 0

    if "observation_ts_clean" in df.columns:
        df["obs_day_of_week"] = df["observation_ts_clean"].dt.dayofweek.fillna(0).astype(int)
        df["obs_hour"] = df["observation_ts_clean"].dt.hour.fillna(0).astype(int)
    else:
        df["obs_day_of_week"] = 0
        df["obs_hour"] = 0

    df["is_weekend_departure"] = df["departure_day_of_week"].isin([5, 6]).astype(int)
    df["days_squared"] = df["days_before_departure"] ** 2
    df["days_log"] = np.log1p(np.clip(df["days_before_departure"], a_min=0, a_max=None))

    # numeric features
    df["snapshot_number"] = pd.to_numeric(df["snapshot_number"], errors="coerce").fillna(0)

    # raw categorical features for CatBoost
    df["airline_cat"] = df["airline_clean"].astype(str)
    df["flight_number_cat"] = df["flight_number_clean"].astype(str)
    df["origin_cat"] = df["origin_clean"].astype(str)
    df["destination_cat"] = df["destination_clean"].astype(str)
    df["origin_airport_cat"] = df["origin_airport_clean"].astype(str)
    df["destination_airport_cat"] = df["destination_airport_clean"].astype(str)
    df["direction_cat"] = df["direction"].astype(str)

    # encoded versions for LR / RF
    enc_cols = [
        "airline_cat",
        "flight_number_cat",
        "origin_cat",
        "destination_cat",
        "origin_airport_cat",
        "destination_airport_cat",
        "direction_cat",
    ]
    for col in enc_cols:
        df[col + "_enc"] = df[col].astype("category").cat.codes

    return df


def get_numeric_feature_columns():
    return [
        "days_before_departure",
        "departure_day_of_week",
        "departure_month",
        "departure_day",
        "departure_hour",
        "obs_day_of_week",
        "obs_hour",
        "is_weekend_departure",
        "days_squared",
        "days_log",
        "snapshot_number",
        "airline_cat_enc",
        "flight_number_cat_enc",
        "origin_cat_enc",
        "destination_cat_enc",
        "origin_airport_cat_enc",
        "destination_airport_cat_enc",
        "direction_cat_enc",
    ]


def get_catboost_feature_columns():
    return [
        "days_before_departure",
        "departure_day_of_week",
        "departure_month",
        "departure_day",
        "departure_hour",
        "obs_day_of_week",
        "obs_hour",
        "is_weekend_departure",
        "days_squared",
        "days_log",
        "snapshot_number",
        "airline_cat",
        "flight_number_cat",
        "origin_cat",
        "destination_cat",
        "origin_airport_cat",
        "destination_airport_cat",
        "direction_cat",
    ]


def get_catboost_cat_features():
    return [
        "airline_cat",
        "flight_number_cat",
        "origin_cat",
        "destination_cat",
        "origin_airport_cat",
        "destination_airport_cat",
        "direction_cat",
    ]


# =========================================================
# 5. SPLIT
# =========================================================
def chronological_split(df):
    n = len(df)
    train_end = int(n * 0.60)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


# =========================================================
# 6. CV HELPERS
# =========================================================
def evaluate_cv_linear(X_train, y_train, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []

    for tr_idx, va_idx in tscv.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        model = LinearRegression()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        fold_scores.append(calc_metrics(y_va, pred))

    return {
        "cv_mae": float(np.mean([m["mae"] for m in fold_scores])),
        "cv_rmse": float(np.mean([m["rmse"] for m in fold_scores])),
        "cv_r2": float(np.mean([m["r2"] for m in fold_scores])),
    }


def evaluate_cv_rf(X_train, y_train, param_grid, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_score = np.inf
    best_params = None
    best_cv = None

    for params in ParameterGrid(param_grid):
        fold_metrics = []

        for tr_idx, va_idx in tscv.split(X_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            model = RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **params
            )
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)
            fold_metrics.append(calc_metrics(y_va, pred))

        mean_mae = float(np.mean([m["mae"] for m in fold_metrics]))
        if mean_mae < best_score:
            best_score = mean_mae
            best_params = params
            best_cv = {
                "cv_mae": mean_mae,
                "cv_rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
                "cv_r2": float(np.mean([m["r2"] for m in fold_metrics])),
            }

    return best_params, best_cv


def evaluate_cv_catboost(X_train, y_train, feature_cols, cat_features, param_grid, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_score = np.inf
    best_params = None
    best_cv = None

    for params in ParameterGrid(param_grid):
        fold_metrics = []

        for tr_idx, va_idx in tscv.split(X_train):
            X_tr, X_va = X_train.iloc[tr_idx][feature_cols], X_train.iloc[va_idx][feature_cols]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            model = CatBoostRegressor(
                iterations=500,
                loss_function="RMSE",
                verbose=False,
                random_state=RANDOM_STATE,
                early_stopping_rounds=50,
                use_best_model=True,
                **params
            )

            model.fit(
                X_tr, y_tr,
                eval_set=(X_va, y_va),
                cat_features=cat_features,
                verbose=False
            )

            pred = model.predict(X_va)
            fold_metrics.append(calc_metrics(y_va, pred))

        mean_mae = float(np.mean([m["mae"] for m in fold_metrics]))
        if mean_mae < best_score:
            best_score = mean_mae
            best_params = params
            best_cv = {
                "cv_mae": mean_mae,
                "cv_rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
                "cv_r2": float(np.mean([m["r2"] for m in fold_metrics])),
            }

    return best_params, best_cv


# =========================================================
# 7. TRAIN ROUTE
# =========================================================
def train_route_models(route_data, route_key, save_dir="models_tabular_final"):
    if route_key not in route_data:
        print(f"❌ Маршрут {route_key} не найден")
        return None

    df = route_data[route_key]["data"].copy()
    config = route_data[route_key]["config"]

    df = prepare_features(df)
    df = df.dropna(subset=["price"])
    df = df.sort_values("observation_ts_clean").reset_index(drop=True)

    if len(df) < MIN_ROUTE_ROWS:
        print(f"⚠️ Недостаточно данных для обучения: {len(df)}")
        return None

    train_df, val_df, test_df = chronological_split(df)

    num_features = get_numeric_feature_columns()
    catboost_features = get_catboost_feature_columns()
    cat_features = get_catboost_cat_features()

    X_train_num = train_df[num_features].fillna(0)
    X_val_num = val_df[num_features].fillna(0)
    X_test_num = test_df[num_features].fillna(0)

    y_train = train_df["price"]
    y_val = val_df["price"]
    y_test = test_df["price"]

    X_train_cat = train_df[catboost_features].copy()
    X_val_cat = val_df[catboost_features].copy()
    X_test_cat = test_df[catboost_features].copy()

    os.makedirs(save_dir, exist_ok=True)

    print(f"\n📊 Данные для обучения ({config['name']}):")
    print(f"   Всего записей: {len(df)}")
    print(f"   Train: {len(train_df)}")
    print(f"   Validation: {len(val_df)}")
    print(f"   Test: {len(test_df)}")

    results = {}

    # -------------------------------
    # Linear Regression
    # -------------------------------
    print("\n🔹 Linear Regression...")
    lr_cv = evaluate_cv_linear(X_train_num, y_train, n_splits=3)

    lr_model = LinearRegression()
    lr_model.fit(X_train_num, y_train)

    lr_pred_train = lr_model.predict(X_train_num)
    lr_pred_val = lr_model.predict(X_val_num)
    lr_pred_test = lr_model.predict(X_test_num)

    results["LinearRegression"] = {
        "model": lr_model,
        "train_metrics": calc_metrics(y_train, lr_pred_train),
        "val_metrics": calc_metrics(y_val, lr_pred_val),
        "test_metrics": calc_metrics(y_test, lr_pred_test),
        "cv_metrics": lr_cv,
        "coefficients": dict(zip(num_features, lr_model.coef_)),
        "intercept": float(lr_model.intercept_)
    }

    joblib.dump(lr_model, os.path.join(save_dir, f"{route_key}_linear.joblib"))

    # -------------------------------
    # Random Forest
    # -------------------------------
    print("🔹 Random Forest...")
    rf_best_params, rf_cv = evaluate_cv_rf(X_train_num, y_train, RF_PARAM_GRID, n_splits=3)

    rf_model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **rf_best_params
    )
    rf_model.fit(X_train_num, y_train)

    rf_pred_train = rf_model.predict(X_train_num)
    rf_pred_val = rf_model.predict(X_val_num)
    rf_pred_test = rf_model.predict(X_test_num)

    results["RandomForest"] = {
        "model": rf_model,
        "best_params": rf_best_params,
        "train_metrics": calc_metrics(y_train, rf_pred_train),
        "val_metrics": calc_metrics(y_val, rf_pred_val),
        "test_metrics": calc_metrics(y_test, rf_pred_test),
        "cv_metrics": rf_cv,
        "feature_importance": dict(zip(num_features, rf_model.feature_importances_))
    }

    joblib.dump(rf_model, os.path.join(save_dir, f"{route_key}_randomforest.joblib"))

    # -------------------------------
    # CatBoost
    # -------------------------------
    if CATBOOST_AVAILABLE:
        print("🔹 CatBoost...")
        cat_best_params, cat_cv = evaluate_cv_catboost(
            X_train_cat, y_train,
            feature_cols=catboost_features,
            cat_features=cat_features,
            param_grid=CAT_PARAM_GRID,
            n_splits=3
        )

        cat_model = CatBoostRegressor(
            iterations=500,
            loss_function="RMSE",
            verbose=False,
            random_state=RANDOM_STATE,
            early_stopping_rounds=50,
            use_best_model=True,
            **cat_best_params
        )

        cat_model.fit(
            X_train_cat, y_train,
            eval_set=(X_val_cat, y_val),
            cat_features=cat_features,
            verbose=False
        )

        cat_pred_train = cat_model.predict(X_train_cat)
        cat_pred_val = cat_model.predict(X_val_cat)
        cat_pred_test = cat_model.predict(X_test_cat)

        results["CatBoost"] = {
            "model": cat_model,
            "best_params": cat_best_params,
            "train_metrics": calc_metrics(y_train, cat_pred_train),
            "val_metrics": calc_metrics(y_val, cat_pred_val),
            "test_metrics": calc_metrics(y_test, cat_pred_test),
            "cv_metrics": cat_cv,
            "feature_importance": dict(zip(catboost_features, cat_model.get_feature_importance(
                Pool(X_train_cat, y_train, cat_features=cat_features)
            )))
        }

        cat_model.save_model(os.path.join(save_dir, f"{route_key}_catboost.cbm"))
    else:
        results["CatBoost"] = None
        print("⚠️ CatBoost пропущен")

    metadata = {
        "route_key": route_key,
        "route_name": config["name"],
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "numeric_features": num_features,
        "catboost_features": catboost_features,
        "cat_features": cat_features,
        "train_date": datetime.now().isoformat(),
        "results": {
            model_name: {
                k: v for k, v in model_data.items() if k != "model"
            }
            for model_name, model_data in results.items()
            if model_data is not None
        }
    }
    joblib.dump(metadata, os.path.join(save_dir, f"{route_key}_metadata.joblib"))

    return {
        "results": results,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "num_features": num_features,
        "catboost_features": catboost_features,
    }


# =========================================================
# 8. PRINTING
# =========================================================
def print_route_results(route_key, route_name, train_output):
    results = train_output["results"]

    print("\n" + "=" * 90)
    print(f"📋 РЕЗУЛЬТАТЫ: {route_name}")
    print("=" * 90)

    rows = []
    for model_name, model_data in results.items():
        if model_data is None:
            continue

        rows.append({
            "Model": model_name,
            "CV MAE": round(model_data["cv_metrics"]["cv_mae"], 2),
            "CV RMSE": round(model_data["cv_metrics"]["cv_rmse"], 2),
            "CV R²": round(model_data["cv_metrics"]["cv_r2"], 4),

            "Train MAE": round(model_data["train_metrics"]["mae"], 2),
            "Train RMSE": round(model_data["train_metrics"]["rmse"], 2),
            "Train R²": round(model_data["train_metrics"]["r2"], 4),

            "Val MAE": round(model_data["val_metrics"]["mae"], 2),
            "Val RMSE": round(model_data["val_metrics"]["rmse"], 2),
            "Val R²": round(model_data["val_metrics"]["r2"], 4),

            "Test MAE": round(model_data["test_metrics"]["mae"], 2),
            "Test RMSE": round(model_data["test_metrics"]["rmse"], 2),
            "Test R²": round(model_data["test_metrics"]["r2"], 4),
        })

    print(pd.DataFrame(rows).to_string(index=False))

    best_model = None
    best_test_mae = np.inf
    for model_name, model_data in results.items():
        if model_data is None:
            continue
        if model_data["test_metrics"]["mae"] < best_test_mae:
            best_test_mae = model_data["test_metrics"]["mae"]
            best_model = model_name

    if best_model is not None:
        print(f"\n🏆 Лучший по Test MAE: {best_model} ({best_test_mae:.2f} ₽)")


# =========================================================
# 9. MAIN
# =========================================================
def main():
    print("=" * 90)
    print("ТАБЛИЧНЫЕ BASELINE-МОДЕЛИ ДЛЯ ПРОГНОЗА ЦЕН НА АВИАБИЛЕТЫ")
    print("Chronological split: 60% train / 25% validation / 15% test")
    print("CV внутри train: TimeSeriesSplit")
    print("=" * 90)

    df = load_and_prepare_data("combined_prices_for_dates_oneway.csv")
    if df is None or df.empty:
        print("❌ После очистки не осталось данных")
        return None, None

    route_data = filter_routes(df)
    if not route_data:
        print("❌ Маршруты не найдены")
        return None, None

    stats = analyze_route_statistics(route_data)

    print("\nСВОДНАЯ СТАТИСТИКА ПО МАРШРУТАМ:")
    for route_key, s in stats.items():
        print(f"\n✈️ {s['route_name']}")
        print(f"   Наблюдений: {s['total_observations']}")
        print(f"   Цена: {s['price_min']:.0f} - {s['price_max']:.0f} ₽")
        print(f"   Средняя: {s['price_mean']:.2f} ₽")
        print(f"   CV: {s['price_cv']:.2f}%")

    save_dir = "models_tabular_final"
    os.makedirs(save_dir, exist_ok=True)

    all_outputs = {}
    summary_rows = []

    for route_key, info in route_data.items():
        print(f"\n{'=' * 70}")
        print(f"🚀 Обучение маршрута: {info['config']['name']}")
        print(f"{'=' * 70}")

        train_output = train_route_models(route_data, route_key, save_dir=save_dir)
        if train_output is None:
            continue

        all_outputs[route_key] = train_output
        print_route_results(route_key, info["config"]["name"], train_output)

        for model_name, model_data in train_output["results"].items():
            if model_data is None:
                continue
            summary_rows.append({
                "route_key": route_key,
                "route_name": info["config"]["name"],
                "model": model_name,
                "cv_mae": model_data["cv_metrics"]["cv_mae"],
                "cv_rmse": model_data["cv_metrics"]["cv_rmse"],
                "cv_r2": model_data["cv_metrics"]["cv_r2"],
                "train_mae": model_data["train_metrics"]["mae"],
                "train_rmse": model_data["train_metrics"]["rmse"],
                "train_r2": model_data["train_metrics"]["r2"],
                "val_mae": model_data["val_metrics"]["mae"],
                "val_rmse": model_data["val_metrics"]["rmse"],
                "val_r2": model_data["val_metrics"]["r2"],
                "test_mae": model_data["test_metrics"]["mae"],
                "test_rmse": model_data["test_metrics"]["rmse"],
                "test_r2": model_data["test_metrics"]["r2"],
            })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(save_dir, "all_routes_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 90)
        print("📊 ОБЩАЯ СВОДКА")
        print("=" * 90)
        print(summary_df.sort_values(["route_name", "test_mae"]).to_string(index=False))
        print(f"\n📄 Сводка сохранена: {summary_path}")

    print("\n" + "=" * 90)
    print("✅ АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 90)

    return stats, all_outputs


if __name__ == "__main__":
    stats, all_outputs = main()
