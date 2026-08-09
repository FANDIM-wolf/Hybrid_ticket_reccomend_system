"""
Загрузчик snapshot-файлов.
Находит все snapshot*.csv в DATA_DIR, выбирает те, что относятся к текущему
DATA_MODE (synthetic или real), и возвращает объединённый DataFrame.
Структура директорий:
data/
├── synthetic/oneway/snapshot_synthetic_oneway_.csv
└── prices_for_dates/2026-04/oneway/MOW_AER/snapshot_.csv   ← от реального парсера
"""
from __future__ import annotations
import glob
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import pandas as pd
from app.config import DATA_DIR, DATA_MODE

logger = logging.getLogger(__name__)

def find_snapshot_files(mode: Optional[str] = None) -> List[Path]:
    """
    Ищет snapshot CSV в зависимости от режима.
    mode='synthetic' → только файлы из data/synthetic/
    mode='real'      → только файлы из prices_for_dates (как в исходном export_old.py)
    """
    mode = mode or DATA_MODE
    if mode == "synthetic":
        pattern = str(DATA_DIR / "synthetic" / "**" / "snapshot*.csv")
    elif mode == "real":
        pattern = str(DATA_DIR / "prices_for_dates" / "**" / "snapshot*.csv")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    files = [Path(f) for f in glob.glob(pattern, recursive=True)]
    # Фильтрация: отбрасываем raw_ (сырые до нормализации) и get_latest_prices
    files = [f for f in files if "snapshot_raw_" not in f.name
             and "get_latest_prices" not in str(f)]
    logger.info(f"Found {len(files)} snapshot files (mode={mode})")
    return files

def extract_date_from_filename(filepath: Path) -> Optional[datetime]:
    """Извлекает штамп времени из имени файла (используется для хронологической сортировки)."""
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{6}Z).csv$", filepath.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%dT%H%M%SZ")
    except ValueError:
        return None

def load_all_snapshots(files: List[Path]) -> pd.DataFrame:
    """Загружает и объединяет все snapshot-файлы в один DataFrame."""
    if not files:
        logger.warning("No snapshot files to load")
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False, on_bad_lines="skip")
            frames.append(df)
        except Exception as e:
            logger.error(f"Failed to read {f}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(combined)} rows from {len(files)} files")
    return combined