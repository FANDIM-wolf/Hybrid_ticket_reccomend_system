"""
Генератор синтетических данных, адаптированный для прототипа СППР.

Ключевое отличие от исходного generate_synthetic.py: результат не сохраняется
одним файлом synthetic_lstm.csv, а раскладывается на snapshot-файлы в формате,
совместимом с ETL-пайплайном. Это обеспечивает единый путь данных:
    parser → snapshot CSV → ETL → SQLite
независимо от того, синтетика это или реальные данные из Travelpayouts.

Параметры профилей, формулы U-кривой, календарные эффекты и генерация цены
полностью воспроизводят исходный скрипт НИР — изменена только выходная форма.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from app.config import (
    DATA_DIR,
    SYNTH_N_ROUTES,
    SYNTH_N_DAYS,
    SYNTH_OBS_PER_DAY,
    SYNTH_SEED,
    SYNTH_START_DATE,
)

logger = logging.getLogger(__name__)

# ============================================================
# КОНФИГУРАЦИЯ ГЕНЕРАЦИИ
# ============================================================
# Профили повторяют кластеризацию авиакомпаний из раздела 2.2 НИР
PROFILES: Dict[str, dict] = {
    "aeroflot_like": {
        "base_price_mean": 22000, "base_price_std": 3000,
        "ar_coef": 0.50, "noise_sigma": 0.005,
        "jump_prob": 0.18,  # ↑ было 0.11 → больше тяжёлых хвостов
        "duration_range": (180, 480),
    },
    "utair_like": {
        "base_price_mean": 14000, "base_price_std": 2500,
        "ar_coef": 0.45, "noise_sigma": 0.007,
        "jump_prob": 0.20,  # ↑ было 0.15
        "duration_range": (150, 420),
    },
    "pobeda_like": {
        "base_price_mean": 8000, "base_price_std": 1500,
        "ar_coef": 0.40, "noise_sigma": 0.010,
        "jump_prob": 0.22,  # ↑ было 0.20
        "duration_range": (120, 360),
    },
}
SHOCK_CLIP = 0.25

DIRECTIONS = ["MOW-AER", "MOW-LED", "MOW-KZN", "MOW-OVB", "MOW-KRR"]

HOLIDAYS = {
    datetime(2025, 12, 31).date(), datetime(2026, 1, 1).date(),
    datetime(2026, 1, 2).date(), datetime(2026, 1, 7).date(),
    datetime(2026, 2, 23).date(), datetime(2026, 3, 8).date(),
}


# ============================================================
# ПРАВИЛА ИЗ НИР
# ============================================================
def u_curve_multiplier(d: int, rng: np.random.Generator) -> float:
    """U-образная кривая глубины бронирования (раздел 1.2 НИР)."""
    if d >= 60:
        return 1.25 + 0.002 * (d - 60)
    elif d >= 30:
        return 0.85 + (1.25 - 0.85) * (d - 30) / 30
    elif d >= 14:
        return 0.85 + (1.05 - 0.85) * (30 - d) / 16
    elif d >= 4:
        return 1.05 + (1.45 - 1.05) * (14 - d) / 10
    elif d >= 1:
        return 0.90 + rng.uniform(-0.05, 0.10)
    return 1.00


def calendar_multiplier(departure_date: datetime) -> float:
    """Календарные эффекты: выходные, праздники (раздел 1.2 НИР)."""
    mult = 1.0
    dow = departure_date.weekday()
    if dow in (4, 6):     # пятница, воскресенье
        mult *= 1.10
    elif dow == 5:        # суббота
        mult *= 1.04
    if departure_date.date() in HOLIDAYS:
        mult *= 1.28
    return mult


def heavy_tailed_noise(sigma: float, jump_prob: float, rng: np.random.Generator) -> float:
    """Гауссов шум + редкие Student-t скачки. Даёт тяжёлые хвосты (skew≈0.68, kurt≈3.67)."""
    if rng.random() < jump_prob:
        jump = rng.standard_t(df=4) * sigma * 8
        if jump > 0:
            jump *= 1.35
        return jump
    return rng.normal(0, sigma)


# ============================================================
# ГЕНЕРАЦИЯ СЫРОГО РЯДА ДЛЯ ОДНОГО МАРШРУТА
# ============================================================
def generate_route(route_id: int, rng: np.random.Generator) -> List[dict]:
    """Формирует список rows для одного синтетического билета."""
    profile_name = list(PROFILES.keys())[route_id % len(PROFILES)]
    profile = PROFILES[profile_name]
    direction = DIRECTIONS[route_id % len(DIRECTIONS)]
    option_name = f"{profile_name}_{direction}_{route_id}"

    start_date = datetime.strptime(SYNTH_START_DATE, "%Y-%m-%d")
    base_price = max(
        rng.normal(profile["base_price_mean"], profile["base_price_std"]), 3000
    )
    duration = int(rng.integers(*profile["duration_range"]))
    departure_date = start_date + timedelta(days=SYNTH_N_DAYS + 30)
    usd_path = 84.0 + np.cumsum(rng.normal(0, 0.05, SYNTH_N_DAYS))

    rows: List[dict] = []
    prev_price = None
    for day_idx in range(SYNTH_N_DAYS):
        current_date = start_date + timedelta(days=day_idx)
        days_until = (departure_date - current_date).days
        u_mult = u_curve_multiplier(days_until, rng)
        cal_mult = calendar_multiplier(departure_date)
        usd_mult = 1.0 + 0.003 * max(0, usd_path[day_idx] - 84)
        fundamental = base_price * u_mult * cal_mult * usd_mult

        for slot_idx, slot in enumerate(["morning", "evening"][:SYNTH_OBS_PER_DAY]):
            if prev_price is None:
                price = fundamental
            else:
                ar = profile["ar_coef"]
                shock = heavy_tailed_noise(profile["noise_sigma"], profile["jump_prob"], rng)
                shock = float(np.clip(shock, -SHOCK_CLIP, SHOCK_CLIP))
                price = ar * prev_price + (1 - ar) * fundamental
                price = price * (1 + shock)

            rows.append({
                "option_name": option_name,
                "profile": profile_name,
                "direction": direction,
                "observation_date": current_date.date().isoformat(),
                "time_slot": slot,
                "time_encoded": slot_idx,
                "days_until_departure": int(days_until),
                "day_of_week": current_date.weekday(),
                "best_price": round(float(price), 2),
                "flight_duration_minutes": duration,
                "usd_rate": round(float(usd_path[day_idx]), 4),
                # эти поля нужны для совместимости с реальным снапшотом парсера
                "ticket_key": option_name,       # для синтетики option_name == ticket_key
                "airline": profile_name.split("_")[0].upper(),
                "flight_number": f"{route_id:04d}",
            })
            prev_price = price
    return rows


# ============================================================
# ЗАПИСЬ В SNAPSHOT CSV
# ============================================================
def write_snapshots(all_rows: List[dict]) -> Path:
    """
    Пишет сгенерированные строки в один snapshot CSV.

    Имя файла имитирует формат парсера: snapshot_synthetic_oneway_{stamp}.csv
    Это позволяет ETL обрабатывать синтетические и реальные данные
    одной и той же процедурой find_snapshot_files.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out_dir = DATA_DIR / "synthetic" / "oneway"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"snapshot_synthetic_oneway_{stamp}.csv"

    df = pd.DataFrame(all_rows)
    df["data_source"] = "synthetic"
    df["run_at_utc"] = datetime.now(timezone.utc).isoformat()
    df["observation_ts"] = df["observation_date"] + "T" + df["time_slot"].map(
        {"morning": "08:00:00Z", "evening": "20:00:00Z"}
    )

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Synthetic snapshot written: {out_path} ({len(df)} rows)")
    return out_path


def _clean_old_synthetic_snapshots() -> None:
    """Удаляет старые синтетические snapshot CSV, чтобы rebuild был идемпотентным."""
    old_dir = DATA_DIR / "synthetic"
    if not old_dir.exists():
        return
    removed = 0
    for f in old_dir.rglob("snapshot*.csv"):
        f.unlink()
        removed += 1
    if removed:
        logger.info(f"Removed {removed} stale synthetic snapshot files")


def generate_full_dataset() -> Path:
    """
    Генерирует полный синтетический датасет и пишет его в DATA_DIR.

    Перед генерацией удаляет старые синтетические snapshot'ы, чтобы
    повторный запуск cold_start --rebuild не приводил к накоплению файлов.

    Возвращает путь к созданному snapshot-файлу.
    Используется в cold_start.py при DATA_MODE='synthetic'.
    """
    _clean_old_synthetic_snapshots()
    rng = np.random.default_rng(SYNTH_SEED)
    logger.info(
        f"Generating synthetic data: {SYNTH_N_ROUTES} routes × "
        f"{SYNTH_N_DAYS} days × {SYNTH_OBS_PER_DAY} obs/day"
    )
    all_rows: List[dict] = []
    for route_id in range(SYNTH_N_ROUTES):
        all_rows.extend(generate_route(route_id, rng))
    logger.info(f"Generated {len(all_rows)} raw rows")
    df = pd.DataFrame(all_rows)

    return write_snapshots(all_rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_full_dataset()
