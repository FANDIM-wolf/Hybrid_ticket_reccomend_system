"""
Cold start прототипа СППР.
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DATA_MODE, DATA_DIR, FINAL_DATASET_NAME  # noqa: E402
from app.db import init_db, count_observations, count_unique_tickets  # noqa: E402
from app.etl.snapshot_loader import find_snapshot_files, load_all_snapshots  # noqa: E402
from app.etl.feature_builder import run_etl_pipeline  # noqa: E402
from app.ml.training import has_active_models, train_all, get_active_models_info  # noqa: E402
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cold_start")

MIN_OBSERVATIONS = 100
MIN_TICKETS = 3

def check_data_integrity() -> bool:
    n_obs = count_observations()
    n_tickets = count_unique_tickets()
    logger.info(f"DB state: {n_obs} observations, {n_tickets} unique tickets")
    if n_obs < MIN_OBSERVATIONS:
        logger.warning(f"Too few observations ({n_obs} < {MIN_OBSERVATIONS})")
        return False
    if n_tickets < MIN_TICKETS:
        logger.warning(f"Too few unique tickets ({n_tickets} < {MIN_TICKETS})")
        return False
    return True

def load_final_dataset_to_db(rebuild: bool) -> int:
    """
    Загружает data/final_data_set_train.csv в observations через ETL.
    """
    csv_path = DATA_DIR / FINAL_DATASET_NAME
    if not csv_path.exists():
        logger.warning(f"Final dataset not found: {csv_path}")
        return 0

    logger.info(f"Loading prepared dataset: {csv_path}")

    # Читаем CSV для предварительного анализа
    df_raw = pd.read_csv(csv_path, encoding="utf-8-sig", nrows=5)

    # Полная загрузка для ETL
    df_full = pd.read_csv(csv_path, encoding="utf-8-sig")

    # ============================================================
    # НОРМАЛИЗАЦИЯ КОЛОНОК (как в training.py)
    # ============================================================
    logger.info("Normalizing column names...")

    # date → observation_date
    if "observation_date" not in df_full.columns and "date" in df_full.columns:
        df_full["observation_date"] = df_full["date"]
        logger.info("  Mapped 'date' → 'observation_date'")

    # time → time_slot
    if "time_slot" not in df_full.columns and "time" in df_full.columns:
        df_full["time_slot"] = df_full["time"]
        logger.info("  Mapped 'time' → 'time_slot'")

    # option → profile (если нужно)
    if "profile" not in df_full.columns and "option" in df_full.columns:
        # Не переопределяем, если option используется для option_name
        logger.info("  'option' column preserved for option_name generation")

    # ticket_key
    if "ticket_key" not in df_full.columns:
        df_full["ticket_key"] = df_full.get("option_name", df_full.get("direction", "unknown"))

    # ============================================================
    # АНАЛИЗ ЗАГРУЖЕННЫХ ДАННЫХ
    # ============================================================
    logger.info("=" * 60)
    logger.info("DATASET ANALYSIS")
    logger.info(f"Total rows: {len(df_full)}")
    logger.info(f"Columns found ({len(df_full.columns)}): {list(df_full.columns)[:20]}...")

    # Проверка наличия ключевых колонок ПОСЛЕ нормализации
    required_cols = ["direction", "option", "best_price", "observation_date"]
    missing = [c for c in required_cols if c not in df_full.columns]
    if missing:
        logger.error(f"CRITICAL: Missing required columns after normalization: {missing}")
        return 0

    # Генерация option_name если нет
    if "option_name" not in df_full.columns:
        logger.info("Generating numbered 'option_name' for final dataset")
        # Базовое имя варианта билета: направление + опция
        _base = df_full["direction"].astype(str) + "_" + df_full["option"].astype(str)
        # Детерминированный порядковый номер по ОТСОРТИРОВАННОМУ списку
        # уникальных имён: не зависит от порядка строк, поэтому
        # cold_start --rebuild каждый раз присваивает ОДНИ И ТЕ ЖЕ номера.
        _unique = sorted(_base.unique())
        _width = max(2, len(str(len(_unique))))
        _id_map = {name: f"_T{i:0{_width}d}" for i, name in enumerate(_unique, start=1)}
        df_full["option_name"] = _base + _base.map(_id_map)

    unique_options = df_full["option_name"].nunique()
    unique_directions = df_full["direction"].nunique()

    # Проверяем, есть ли данные о рейсах
    flight_cols = [c for c in df_full.columns if c.startswith("outbound_") or c.startswith("return_")]
    flight_cols_filled = [c for c in flight_cols if df_full[c].notna().any()]

    logger.info(f"Unique directions: {unique_directions} ({df_full['direction'].unique().tolist()})")
    logger.info(f"Unique tickets (option_name): {unique_options}")
    logger.info(f"Flight detail columns: {len(flight_cols)} total, {len(flight_cols_filled)} with data")
    if flight_cols_filled:
        logger.info(f"  Filled: {flight_cols_filled[:5]}...")
    logger.info(f"Sample options: {df_full['option_name'].head(5).tolist()}")
    logger.info("=" * 60)

    # ETL recalculate features and load to DB
    n = run_etl_pipeline(df_full, data_source="prepared_csv", truncate=rebuild)
    return n

def ingest_data(rebuild: bool = False) -> int:
    """
    Загружает данные.
    """
    # 1. Сначала пытаемся загрузить final_data_set_train.csv
    final_csv = DATA_DIR / FINAL_DATASET_NAME
    if final_csv.exists():
        logger.info(f"Found prepared dataset: {final_csv}")
        n = load_final_dataset_to_db(rebuild=rebuild)
        if n > 0:
            return n
        logger.warning("Prepared CSV load failed, falling back to DATA_MODE")

    # 2. fallback: синтетика или реал
    if DATA_MODE == "synthetic":
        logger.info("DATA_MODE=synthetic → generating synthetic snapshots")
        from app.synthetic.generator import generate_full_dataset
        generate_full_dataset()
    elif DATA_MODE == "real":
        logger.info("DATA_MODE=real → expecting snapshot CSV in data/prices_for_dates/")
        files = find_snapshot_files(mode="real")
        if not files:
            logger.error("No snapshot files found. Run parser first.")
            return 0
    else:
        raise ValueError(f"Unknown DATA_MODE: {DATA_MODE}")

    files = find_snapshot_files()
    if not files:
        logger.error("ETL: no snapshot files to process")
        return 0

    df_raw = load_all_snapshots(files)
    if df_raw.empty:
        logger.error("ETL: loaded empty DataFrame")
        return 0

    return run_etl_pipeline(df_raw, data_source=DATA_MODE, truncate=rebuild)

def _log_training_metrics(result: dict) -> None:
    """Печать ключевых метрик после обучения."""
    logger.info(f"Training finished in {result['duration_seconds']:.1f}s")
    # ... (ваш код логирования метрик) ...

def main():
    parser = argparse.ArgumentParser(description="Cold start for DSS prototype")
    parser.add_argument("--rebuild", action="store_true", help="Force full rebuild")
    parser.add_argument("--check-only", action="store_true", help="Only check integrity")
    parser.add_argument("--no-serve", action="store_true", help="Do not start uvicorn")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DSS prototype cold start")
    logger.info(f"DATA_MODE={DATA_MODE}  rebuild={args.rebuild}")
    logger.info("=" * 60)

    # Step 1
    logger.info("Step 1: initializing database schema")
    init_db()

    # Step 2
    logger.info("Step 2: checking data integrity")
    data_ok = check_data_integrity()

    if args.check_only:
        logger.info(f"Check-only mode: data_ok={data_ok}")
        return 0 if data_ok else 1

    # Step 3
    if args.rebuild or not data_ok:
        reason = "rebuild flag" if args.rebuild else "insufficient data"
        logger.info(f"Step 3: ingesting data (reason: {reason})")
        n = ingest_data(rebuild=args.rebuild)
        if n == 0:
            logger.error("Data ingest failed")
            return 1
        if not check_data_integrity():
            logger.error("Data integrity still insufficient after ingest")
            return 1
    else:
        logger.info("Step 3: skipping ingest, data is sufficient")

    # Step 4: модели
    logger.info("Step 4: checking trained models")
    if args.rebuild or not has_active_models():
        reason = "rebuild flag" if args.rebuild else "no active models"
        logger.info(f"Training LSTM v2 + CatBoost Ensemble (reason: {reason})")
        try:
            result = train_all()
            _log_training_metrics(result)
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            return 1
    else:
        info = get_active_models_info()
        logger.info("Active models found, skipping training:")
        for name, data in info.items():
            logger.info(f"  {name} v{data['version']}: {data['metric_name']}={data['metric_value']:.4f}")

    # Step 5: uvicorn
    if args.no_serve:
        logger.info("Cold start complete (--no-serve). Skipping uvicorn.")
        return 0

    logger.info("Step 5: starting FastAPI on uvicorn")
    from app.config import API_HOST, API_PORT
    import uvicorn
    logger.info(f"Serving at http://{API_HOST}:{API_PORT}")
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, log_level="info")
    return 0

if __name__ == "__main__":
    sys.exit(main())