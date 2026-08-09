"""
Схема базы данных SQLite и helper для подключений.
Таблицы:
  observations      — наблюдения после ETL (ticket_key, время, цена, фичи)
  predictions       — прогнозы моделей (LSTM регрессия + CatBoost классификация)
  residuals         — остатки LSTM и признаки ошибок CatBoost (источник для детектора дрейфа)
  retrain_events    — журнал переобучений (конфиг, метрики до/после, время)
  bandit_state      — сериализованное состояние LinUCB (матрицы A и векторы b для каждой руки)
  models_meta       — метаданные текущих обученных моделей
  recommendations   — рекомендации buy/wait/neutral, выданные пользователю
  decision_logs     — полный лог обращений пользователя к /recommend
  control_logs      — журнал работы модуля управления (триггеры, решения бандита, дрейф)
Используется стандартный sqlite3, без ORM. Для прототипа этого достаточно.
"""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from app.config import DB_PATH

# Схема БД одним блоком. Используется в init_db().
SCHEMA_SQL = """
-- ============================================================
-- observations: нормализованные наблюдения после ETL
-- ============================================================
CREATE TABLE IF NOT EXISTS observations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_key          TEXT NOT NULL,
    option_name         TEXT NOT NULL,
    profile             TEXT,
    direction           TEXT NOT NULL,
    observation_date    TEXT NOT NULL,
    time_slot           TEXT NOT NULL,
    time_encoded        INTEGER NOT NULL,
    days_until_departure INTEGER,
    day_of_week         INTEGER,
    best_price          REAL NOT NULL,
    flight_duration_minutes INTEGER,
    usd_rate            REAL,
    
    -- сгенерированные фичи
    price_change_percent REAL,
    prev_price_change   REAL,
    day_of_week_sin     REAL,
    day_of_week_cos     REAL,
    price_lag_1         REAL,
    price_lag_2         REAL,
    price_lag_3         REAL,
    rolling_mean_3d     REAL,
    rolling_std_3d      REAL,
    price_momentum      REAL,
    price_acceleration  REAL,
    ema_3_day           REAL,
    ema_7_day           REAL,
    rsi_7               REAL,
    jump_class          INTEGER,
    
    -- детали исходящего рейса (outbound)
    outbound_flight         TEXT,
    outbound_aircraft       TEXT,
    outbound_depart_time    TEXT,
    outbound_arrival_time   TEXT,
    outbound_duration_min   INTEGER,
    outbound_depart_airport TEXT,
    outbound_arrival_airport TEXT,
    outbound_date           TEXT,

    -- детали обратного рейса (return)
    return_flight           TEXT,
    return_aircraft         TEXT,
    return_depart_time      TEXT,
    return_arrival_time     TEXT,
    return_duration_min     INTEGER,
    return_depart_airport   TEXT,
    return_arrival_airport  TEXT,
    return_date             TEXT,

    -- служебные
    data_source         TEXT NOT NULL DEFAULT 'synthetic',
    inserted_at         TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_obs_ticket ON observations(ticket_key);
CREATE INDEX IF NOT EXISTS idx_obs_date   ON observations(observation_date);
CREATE INDEX IF NOT EXISTS idx_obs_option ON observations(option_name);

-- ============================================================
-- predictions: прогнозы моделей для конкретной (option, момент времени)
-- ============================================================
CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    option_name         TEXT NOT NULL,
    predicted_at        TEXT NOT NULL,     -- когда сделан прогноз
    target_time         TEXT NOT NULL,     -- момент, для которого прогноз
    current_price       REAL NOT NULL,
    lstm_pred_ratio     REAL,              -- прогноз отношения p[t+2]/p[t]
    lstm_pred_rub       REAL,              -- восстановленный прогноз в рублях
    catboost_class      INTEGER,
    catboost_proba_0    REAL,
    catboost_proba_1    REAL,
    catboost_proba_2    REAL,
    actual_price        REAL,              -- заполняется позднее, когда факт известен
    model_version       TEXT,              -- какая версия модели делала прогноз
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_pred_option ON predictions(option_name);
CREATE INDEX IF NOT EXISTS idx_pred_target ON predictions(target_time);

-- ============================================================
-- residuals: остатки прогнозов, источник для детектора дрейфа
-- ============================================================
CREATE TABLE IF NOT EXISTS residuals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id       INTEGER,
    option_name         TEXT NOT NULL,
    observation_date    TEXT NOT NULL,
    lstm_residual       REAL,              -- Δp_fact − Δp_pred
    catboost_correct    INTEGER,           -- 0 / 1
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);

-- ============================================================
-- retrain_events: журнал переобучений базовых моделей
-- ============================================================
CREATE TABLE IF NOT EXISTS retrain_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    triggered_at        TEXT NOT NULL DEFAULT (datetime('now')),
    trigger_reason      TEXT NOT NULL,     -- 'cold_start' | 'drift' | 'manual'
    config_json         TEXT NOT NULL,     -- выбранная Головой 2 конфигурация
    lstm_mae_before     REAL,
    lstm_mae_after      REAL,
    catboost_f1_before  REAL,
    catboost_f1_after   REAL,
    reward_head1        REAL,              -- R_1 = ΔMetric − λ·C
    reward_head2        REAL,              -- R_2 = ΔMetric
    duration_seconds    REAL
);

-- ============================================================
-- bandit_state: сериализованные Aₐ и b для каждой руки каждой головы
-- ============================================================
CREATE TABLE IF NOT EXISTS bandit_state (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    head                TEXT NOT NULL,     -- 'head1' | 'head2'
    arm_id              TEXT NOT NULL,     -- идентификатор действия
    a_matrix_blob       BLOB NOT NULL,     -- numpy.save() bytes
    b_vector_blob       BLOB NOT NULL,
    n_pulls             INTEGER NOT NULL DEFAULT 0,
    cumulative_reward   REAL NOT NULL DEFAULT 0.0,
    updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (head, arm_id)
);

-- ============================================================
-- models_meta: метаданные текущих обученных моделей
-- ============================================================
CREATE TABLE IF NOT EXISTS models_meta (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name          TEXT NOT NULL,     -- 'lstm' | 'catboost'
    version             TEXT NOT NULL,     -- таймштамп обучения
    model_path          TEXT NOT NULL,     -- ПУТЬ К ФАЙЛУ МОДЕЛИ
    meta_path           TEXT NOT NULL,     -- ПУТЬ К META JSON
    metric_name         TEXT NOT NULL,
    metric_value        REAL NOT NULL,
    trained_at          TEXT NOT NULL DEFAULT (datetime('now')),
    is_active           INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_models_active ON models_meta(model_name, is_active);

-- ============================================================
-- recommendations: выданные пользователю рекомендации
-- ============================================================
CREATE TABLE IF NOT EXISTS recommendations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    option_name         TEXT NOT NULL,
    issued_at           TEXT NOT NULL DEFAULT (datetime('now')),
    current_price       REAL NOT NULL,
    lstm_pred_rub       REAL,
    catboost_class      INTEGER,
    recommendation      TEXT NOT NULL,     -- 'BUY' | 'WAIT' | 'NEUTRAL'
    confidence          REAL,
    reasoning           TEXT               -- человеко-читаемое пояснение
);

-- ============================================================
-- decision_logs: все обращения к /recommend
-- ============================================================
CREATE TABLE IF NOT EXISTS decision_logs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    request_at          TEXT NOT NULL DEFAULT (datetime('now')),
    ticket_key          TEXT,
    option_name         TEXT,
    response_json       TEXT NOT NULL
);

-- ============================================================
-- control_logs: журнал работы модуля управления
-- ============================================================
CREATE TABLE IF NOT EXISTS control_logs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at           TEXT NOT NULL DEFAULT (datetime('now')),
    event_type          TEXT NOT NULL,     -- 'drift_check' | 'head1_decision' | 'head2_decision' | 'retrain_start' | 'retrain_end'
    ph_statistic        REAL,              -- значение Page-Hinkley на момент события
    ddm_error_rate      REAL,              -- текущая частота ошибок CatBoost
    state_vector_json   TEXT,              -- вектор состояния среды
    action_chosen       TEXT,              -- выбранное действие
    expected_reward     REAL,              -- ожидание награды (UCB-score)
    message             TEXT               -- произвольный текст события
);
"""

def init_db() -> None:
    """Создаёт все таблицы, если их нет. Идемпотентна."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        _migrate_add_flight_columns(conn)
    finally:
        conn.close()

def _migrate_add_flight_columns(conn: sqlite3.Connection) -> None:
    """
    Добавляет колонки деталей рейса в существующую таблицу observations.
    Безопасна для повторного вызова — проверяет наличие колонки перед ADD COLUMN.
    """
    new_cols = [
        ("outbound_flight",          "TEXT"),
        ("outbound_aircraft",        "TEXT"),
        ("outbound_depart_time",     "TEXT"),
        ("outbound_arrival_time",    "TEXT"),
        ("outbound_duration_min",    "INTEGER"),
        ("outbound_depart_airport",  "TEXT"),
        ("outbound_arrival_airport", "TEXT"),
        ("outbound_date",            "TEXT"),
        ("return_flight",            "TEXT"),
        ("return_aircraft",          "TEXT"),
        ("return_depart_time",       "TEXT"),
        ("return_arrival_time",      "TEXT"),
        ("return_duration_min",      "INTEGER"),
        ("return_depart_airport",    "TEXT"),
        ("return_arrival_airport",   "TEXT"),
        ("return_date",              "TEXT"),
    ]
    existing = {row[1] for row in conn.execute("PRAGMA table_info(observations)").fetchall()}
    added = []
    for col_name, col_type in new_cols:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE observations ADD COLUMN {col_name} {col_type}")
            added.append(col_name)
    if added:
        conn.commit()
        import logging
        logging.getLogger(__name__).info(f"Migration: added columns {added}")

@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """Контекстный менеджер подключения с row_factory=Row для удобства."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Включаем foreign keys и WAL для параллельного чтения
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
    finally:
        conn.close()

def count_observations() -> int:
    """Сколько наблюдений в БД. Используется cold_start для проверки целостности."""
    with get_conn() as c:
        row = c.execute("SELECT COUNT(*) AS n FROM observations").fetchone()
        return int(row["n"])

def count_unique_tickets() -> int:
    """Сколько уникальных билетов (option_name) в БД."""
    with get_conn() as c:
        row = c.execute("SELECT COUNT(DISTINCT option_name) AS n FROM observations").fetchone()
        return int(row["n"])

# ============================================================
# УПРАВЛЕНИЕ МОДЕЛЯМИ
# ============================================================

def register_model_version(
        model_name: str,
        version: str,
        model_path: str,
        meta_path: str,
        metric_name: str,
        metric_value: float,
) -> None:
    """
    Регистрирует новую версию модели в таблице models_meta.
    Автоматически деактивирует предыдущую активную версию этой модели.
    """
    with get_conn() as c:
        # 1. Деактивируем старую версию
        c.execute(
            "UPDATE models_meta SET is_active = 0 WHERE model_name = ? AND is_active = 1",
            (model_name,),
        )
        # 2. Добавляем новую запись
        c.execute(
            """
            INSERT INTO models_meta 
            (model_name, version, model_path, meta_path, metric_name, metric_value, trained_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'), 1)
            """,
            (model_name, version, model_path, meta_path, metric_name, metric_value),
        )
        c.commit()

def set_active_model(model_name: str, version: str) -> None:
    """
    Явно устанавливает указанную версию модели как активную.
    Используется, если нужно переключиться на конкретную версию без регистрации новой.
    """
    with get_conn() as c:
        # Сначала деактивируем все версии этой модели
        c.execute(
            "UPDATE models_meta SET is_active = 0 WHERE model_name = ?",
            (model_name,),
        )
        # Активируем нужную
        c.execute(
            "UPDATE models_meta SET is_active = 1 WHERE model_name = ? AND version = ?",
            (model_name, version),
        )
        c.commit()

if __name__ == "__main__":
    # Ручной запуск: python -m app.db
    init_db()
    print(f"DB initialized at {DB_PATH}")
    print(f"Observations: {count_observations()}")
    print(f"Unique tickets: {count_unique_tickets()}")