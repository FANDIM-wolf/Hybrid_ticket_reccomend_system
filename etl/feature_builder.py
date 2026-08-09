"""
Построитель признаков и загрузчик в SQLite.
ВСЕ исторические признаки считаются по p_hist = p.shift(1),
чтобы не допустить утечки таргета. Это критично для ML.
"""
from __future__ import annotations
import logging
from typing import List
import numpy as np
import pandas as pd
from app.config import JUMP_THRESHOLD_STABLE, JUMP_THRESHOLD_HIGH
from app.db import get_conn

logger = logging.getLogger(__name__)

REQUIRED_INPUT_COLUMNS = [
    "option_name", "observation_date", "time_encoded",
    "best_price", "day_of_week", "days_until_departure",
]

def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет все производные признаки поверх сырых наблюдений.
    Важно: все лаги, скользящие и EMA считаются по p_hist = p.shift(1), т. е.
    "история до текущей точки, не включая её". Без этого фичи раскрывают
    таргет (price_change_percent[t] = price[t]/price[t-1] - 1) и дают утечку.
    """
    _validate_input(df)
    df = df.sort_values(["option_name", "observation_date", "time_encoded"]).reset_index(drop=True)
    out: List[pd.DataFrame] = []

    for _, g in df.groupby("option_name", sort=False):
        g = g.copy().reset_index(drop=True)
        p = g["best_price"]

        # Таргет: процентное изменение цены от предыдущего замера
        g["price_change_percent"] = p.pct_change().fillna(0) * 100

        # История без текущей точки — основа всех лаговых признаков
        p_hist = p.shift(1)

        # Предыдущее изменение цены (известно к моменту t)
        g["prev_price_change"] = g["price_change_percent"].shift(1).fillna(0)

        # Циклические признаки дня недели
        dow = g["day_of_week"]
        g["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
        g["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)

        # Лаги цены
        for lag in (1, 2, 3):
            g[f"price_lag_{lag}"] = p.shift(lag).fillna(p_hist).fillna(p.iloc[0])

        # Скользящие: 6 замеров ≈ 3 дня
        g["rolling_mean_3d"] = p_hist.rolling(6, min_periods=1).mean().fillna(p.iloc[0])
        g["rolling_std_3d"] = p_hist.rolling(6, min_periods=1).std().fillna(0)

        # Моментум и ускорение
        g["price_momentum"] = (p_hist - p.shift(7)).fillna(0)
        g["price_acceleration"] = g["price_momentum"].diff().fillna(0)

        # EMA (считаем по сдвинутой серии, чтобы ema[t] не содержал p[t])
        g["ema_3_day"] = p_hist.ewm(span=6, adjust=False).mean().fillna(p.iloc[0])
        g["ema_7_day"] = p_hist.ewm(span=14, adjust=False).mean().fillna(p.iloc[0])

        # RSI по 7 замерам
        delta = p_hist.diff()
        gain = delta.clip(lower=0).rolling(7, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(7, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        g["rsi_7"] = (100 - 100 / (1 + rs)).fillna(50)

        out.append(g)

    return pd.concat(out, ignore_index=True)

def add_jump_class(df: pd.DataFrame) -> pd.DataFrame:
    """Метка класса скачка: 0 stable, 1 moderate, 2 high."""
    a = df["price_change_percent"].abs()
    df["jump_class"] = np.where(
        a <= JUMP_THRESHOLD_STABLE, 0,
        np.where(a <= JUMP_THRESHOLD_HIGH, 1, 2)
    )
    return df

# ============================================================
# ЗАГРУЗКА В SQLITE
# ============================================================
OBS_COLUMNS = [
    "ticket_key", "option_name", "profile", "direction",
    "observation_date", "time_slot", "time_encoded",
    "days_until_departure", "day_of_week", "best_price",
    "flight_duration_minutes", "usd_rate",
    "price_change_percent", "prev_price_change",
    "day_of_week_sin", "day_of_week_cos",
    "price_lag_1", "price_lag_2", "price_lag_3",
    "rolling_mean_3d", "rolling_std_3d",
    "price_momentum", "price_acceleration",
    "ema_3_day", "ema_7_day", "rsi_7",
    "jump_class", "data_source",
    # Детали рейсов (outbound)
    "outbound_flight", "outbound_aircraft",
    "outbound_depart_time", "outbound_arrival_time",
    "outbound_duration_min", "outbound_depart_airport",
    "outbound_arrival_airport", "outbound_date",
    # Детали рейсов (return)
    "return_flight", "return_aircraft",
    "return_depart_time", "return_arrival_time",
    "return_duration_min", "return_depart_airport",
    "return_arrival_airport", "return_date",
]

def load_to_db(df: pd.DataFrame, data_source: str = "synthetic", truncate: bool = False) -> int:
    """
    Заливает DataFrame в таблицу observations.
    truncate=True — полностью очищает таблицу перед вставкой.
    Возвращает число вставленных строк.
    """
    if df.empty:
        logger.warning("Attempted to load empty DataFrame into DB")
        return 0

    if "ticket_key" not in df.columns:
        df["ticket_key"] = df["option_name"]
    if "time_slot" not in df.columns:
        df["time_slot"] = df["time_encoded"].map({0: "morning", 1: "evening"}).fillna("unknown")
    if "profile" not in df.columns:
        df["profile"] = None
    if "data_source" not in df.columns:
        df["data_source"] = data_source

    missing_cols = [c for c in OBS_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns for DB insert: {missing_cols}")
        for c in missing_cols:
            df[c] = None

    df_to_insert = df[OBS_COLUMNS].copy()

    # КРИТИЧНО: SQLite не принимает np.nan / pd.NaT — заменяем на None
    df_to_insert = df_to_insert.where(df_to_insert.notna(), None)
    for col in df_to_insert.columns:
        if pd.api.types.is_datetime64_any_dtype(df_to_insert[col]):
            df_to_insert[col] = df_to_insert[col].replace({pd.NaT: None})

    placeholders = ", ".join(["?"] * len(OBS_COLUMNS))
    columns_sql = ", ".join(OBS_COLUMNS)
    insert_sql = f"INSERT INTO observations ({columns_sql}) VALUES ({placeholders})"

    rows = [tuple(r) for r in df_to_insert.itertuples(index=False, name=None)]

    with get_conn() as conn:
        if truncate:
            conn.execute("DELETE FROM observations")
            logger.info("observations table truncated")
        conn.executemany(insert_sql, rows)
        conn.commit()

    logger.info(f"Inserted {len(rows)} rows into observations")
    return len(rows)

def run_etl_pipeline(df_raw: pd.DataFrame, data_source: str, truncate: bool = False) -> int:
    """
    Полный ETL-шаг:
    1. Очистка данных (удаление несоответствий маршрутов и выбросов цен)
    2. Построение признаков (add_features)
    3. Классификация скачков (add_jump_class)
    4. Загрузка в БД (load_to_db)
    Возвращает число вставленных строк.
    """
    logger.info(f"ETL pipeline start: {len(df_raw)} raw rows, source={data_source}")
    if df_raw.empty:
        logger.warning("ETL pipeline stopped: Input DataFrame is empty.")
        return 0

    if "option_name" not in df_raw.columns:
        logger.info("Generating 'option_name' column (direction + option)")
        if "direction" in df_raw.columns and "option" in df_raw.columns:
            df_raw["option_name"] = df_raw["direction"].astype(str) + "_" + df_raw["option"].astype(str)
        else:
            logger.error("Cannot generate option_name: missing 'direction' or 'option' columns")
            return 0

    if "ticket_key" not in df_raw.columns:
        df_raw["ticket_key"] = df_raw["option_name"]

    # 1. ОЧИСТКА ДАННЫХ
    from app.etl.data_cleaner import clean_flight_data, remove_price_outliers

    df_clean = clean_flight_data(df_raw)
    df_clean = remove_price_outliers(df_clean)

    if df_clean.empty:
        logger.warning("ETL pipeline stopped: No valid rows left after cleaning.")
        return 0

    # 2. Построение признаков
    try:
        df_feat = add_features(df_clean)
    except Exception as e:
        logger.error(f"Error in add_features: {e}")
        return 0

    # 3. Классификация скачков
    df_jump = add_jump_class(df_feat)

    # 4. Загрузка в БД
    n = load_to_db(df_jump, data_source=data_source, truncate=truncate)

    logger.info(f"ETL pipeline done: {n} rows in DB")
    return n