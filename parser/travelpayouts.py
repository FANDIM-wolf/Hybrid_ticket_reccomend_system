"""
Адаптер парсера Travelpayouts для прототипа СППР.

Упрощённая версия исходного parser.py, сохраняющая ключевые механизмы:
  - session с retry и backoff
  - ticket_key как MD5 от (direction, departure_at, airline, flight_number)
  - запись snapshot CSV в формате, совместимом с ETL
  - адаптивный sleep на основе заголовков rate limit

Парсер используется в двух сценариях:
  1) cold_start при DATA_MODE='real' (если хотите собрать данные сразу)
  2) ручной вызов через эндпоинт POST /parser/run (итерация 3)
"""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import (
    DATA_DIR,
    PARSER_BACKOFF,
    PARSER_BASE_ROUTES,
    PARSER_CURRENCY,
    PARSER_DEPARTURE_MONTH,
    PARSER_MARKET,
    PARSER_MAX_RETRIES,
    PARSER_TIMEOUT,
    TRAVELPAYOUTS_TOKEN,
)

logger = logging.getLogger(__name__)

URL = "https://api.travelpayouts.com/aviasales/v3/prices_for_dates"


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=PARSER_MAX_RETRIES, read=PARSER_MAX_RETRIES, connect=PARSER_MAX_RETRIES,
        backoff_factor=PARSER_BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({
        "X-Access-Token": TRAVELPAYOUTS_TOKEN,
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "dss-prototype-parser/1.0",
    })
    return session


def _hash(parts: List[str]) -> str:
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


def _build_ticket_key(row: dict, origin: str, destination: str) -> str:
    """
    Воспроизводит формулу ticket_key из parser.py v4.0:
    direction|departure_at|airline|flight_number → MD5.
    """
    direction = f"{origin}->{destination}".upper()
    return _hash([
        direction,
        str(row.get("departure_at", "")),
        str(row.get("airline", "")).upper().strip(),
        str(row.get("flight_number", "")),
    ])


def fetch_route(session: requests.Session, origin: str, destination: str) -> List[dict]:
    """Получает все записи по одному направлению с постраничной обработкой."""
    params = {
        "origin": origin,
        "destination": destination,
        "departure_at": PARSER_DEPARTURE_MONTH,
        "market": PARSER_MARKET,
        "currency": PARSER_CURRENCY,
        "one_way": "true",
        "sorting": "price",
        "limit": 1000,
    }
    all_rows: List[dict] = []
    for page in range(1, 51):
        params["page"] = page
        try:
            r = session.get(URL, params=params, timeout=PARSER_TIMEOUT)
            r.raise_for_status()
            data = r.json().get("data", []) or []
        except Exception as e:
            logger.error(f"{origin}->{destination} page {page}: {e}")
            break
        if not data:
            break
        all_rows.extend(data)
        # rate limit guard
        remaining = r.headers.get("X-Rate-Limit-Remaining")
        if remaining is not None and int(remaining or 10) < 5:
            time.sleep(1.5)
        else:
            time.sleep(0.2)
    return all_rows


def save_snapshot(rows: List[dict], origin: str, destination: str) -> Path:
    """Сохраняет ответы API в snapshot CSV в формате, совместимом с ETL."""
    if not rows:
        logger.warning(f"Empty response for {origin}->{destination}, not saving")
        return None

    stamp =  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out_dir = DATA_DIR / "prices_for_dates" / PARSER_DEPARTURE_MONTH / "oneway" / f"{origin}_{destination}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"snapshot_prices_for_dates_oneway_{origin}_{destination}_{stamp}.csv"

    df = pd.DataFrame(rows)

    # Приводим к колонкам, которые ожидает ETL
    df["option_name"] = df.apply(
        lambda r: _build_ticket_key(dict(r), origin, destination), axis=1
    )
    df["ticket_key"] = df["option_name"]
    df["profile"] = "real"
    df["direction"] = f"{origin}-{destination}"
    df["observation_date"] = datetime.now(timezone.utc).date().isoformat()
    df["time_slot"] = "parser_run"
    df["time_encoded"] = 0
    df["data_source"] = "real"
    df["run_at_utc"] = datetime.now(timezone.utc).isoformat()

    # days_until_departure и day_of_week
    df["departure_at"] = pd.to_datetime(df.get("departure_at"), errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    df["days_until_departure"] = (df["departure_at"] - now).dt.days.fillna(0).astype(int)
    df["day_of_week"] = df["departure_at"].dt.dayofweek.fillna(0).astype(int)

    # переименовываем price → best_price (ETL ожидает best_price)
    if "price" in df.columns and "best_price" not in df.columns:
        df["best_price"] = pd.to_numeric(df["price"], errors="coerce")

    # usd_rate и flight_duration_minutes — если их нет, проставляем None/0
    if "usd_rate" not in df.columns:
        df["usd_rate"] = None
    if "flight_duration_minutes" not in df.columns:
        df["flight_duration_minutes"] = df.get("duration", 0)

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved {len(df)} rows to {out_path}")
    return out_path


def run_parser() -> List[Path]:
    """Полный проход по PARSER_BASE_ROUTES. Возвращает список созданных snapshot-файлов."""
    session = _build_session()
    saved_files: List[Path] = []
    for origin, destination in PARSER_BASE_ROUTES:
        logger.info(f"Fetching {origin} -> {destination}")
        rows = fetch_route(session, origin, destination)
        path = save_snapshot(rows, origin, destination)
        if path:
            saved_files.append(path)
    logger.info(f"Parser done: {len(saved_files)} snapshot files saved")
    return saved_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_parser()
