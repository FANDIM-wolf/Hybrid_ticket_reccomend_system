"""
Global configuration of the DSS prototype.

Contains paths, constants, model parameters and data source switch.
All modules import settings ONLY from here so paths are in one place.
"""
from __future__ import annotations

import os
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
# Project root — directory where config.py lives (app/config.py -> ../)
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"              # snapshot CSV (ETL input)
MODELS_DIR = ROOT_DIR / "models"          # saved LSTM and CatBoost
STATIC_DIR = ROOT_DIR / "static"          # (reserved for UI)
DB_PATH = ROOT_DIR / "dss.sqlite"         # main database

# Create directories if missing
for _d in (DATA_DIR, MODELS_DIR, STATIC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATA SOURCE SWITCH
# ============================================================
# "synthetic" — cold_start generates synthetic dataset
# "real"      — cold_start expects snapshot CSV to already exist in DATA_DIR
#               (collected via POST /parser/run or manually)
#
# Can be overridden via DSS_DATA_MODE environment variable.
DATA_MODE: str = os.getenv("DSS_DATA_MODE", "real")

assert DATA_MODE in ("synthetic", "real"), f"Unknown DATA_MODE: {DATA_MODE}"

# ============================================================
# SYNTHETIC GENERATOR PARAMETERS
# ============================================================
SYNTH_N_ROUTES = 20
SYNTH_N_DAYS = 180
SYNTH_OBS_PER_DAY = 2
SYNTH_SEED = 42
SYNTH_START_DATE = "2025-11-01"

# === Training dataset filename ===
# Must be in data/ — copy of train_lstm_final.csv
FINAL_DATASET_NAME = "final_data_set_train.csv"


# ============================================================
# TRAVELPAYOUTS PARSER
# ============================================================
# Token from original parser.py. In production should be read
# from TRAVELPAYOUTS_TOKEN environment variable, see below.
TRAVELPAYOUTS_TOKEN = os.getenv(
    "TRAVELPAYOUTS_TOKEN",
    "2702fc9fc266eb48403095269c9bdec6",
)
PARSER_BASE_ROUTES = [
    # --- Original routes ---
    ("MOW", "AER"),  # Sochi
    ("MOW", "KRR"),  # Krasnodar
    ("MOW", "KZN"),  # Kazan
    ("MOW", "LED"),  # Saint Petersburg
    ("MOW", "NQZ"),  # Astana

    # --- New routes (10 added) ---
    ("MOW", "OVB"),  # Novosibirsk (major hub)
    ("MOW", "SVX"),  # Yekaterinburg (major hub)
    ("MOW", "KGD"),  # Kaliningrad
    ("MOW", "MRV"),  # Mineralnye Vody (resort)
    ("MOW", "IST"),  # Istanbul
    ("MOW", "ALA"),  # Almaty
    ("MOW", "EVN"),  # Yerevan
    ("MOW", "GYD"),  # Baku
    ("MOW", "TAS"),  # Tashkent
    ("MOW", "DXB"),  # Dubai
]
PARSER_DEPARTURE_MONTH = "2026-06"
PARSER_MARKET = "ru"
PARSER_CURRENCY = "rub"
PARSER_TIMEOUT = 60
PARSER_MAX_RETRIES = 3
PARSER_BACKOFF = 0.5

# ============================================================
# ML HYPERPARAMETERS (defaults for first training)
# ============================================================
# === LSTM v2 — training parameters ===
LSTM_SEQ_LEN: int = 14         # window length
LSTM_HORIZON: int = 2          # steps ahead to predict
LSTM_EPOCHS: int = 100
LSTM_BATCH_SIZE: int = 32
LSTM_LEARNING_RATE: float = 1e-3
LSTM_RANDOM_SEED: int = 42

# === Real directions — separated from synthetic ===
REAL_DIRECTIONS = {"MOW-AER", "MOW-KZN", "MOW-LED"}

# CatBoost
CATBOOST_ITERATIONS = 1500
CATBOOST_LEARNING_RATE = 0.03
CATBOOST_DEPTH = 14
CATBOOST_EARLY_STOPPING = 200

# ============================================================
# PRICE JUMP CLASSIFICATION
# ============================================================
JUMP_THRESHOLD_STABLE: float = 0.1
JUMP_THRESHOLD_HIGH: float = 3.0
JUMP_CLASS_NAMES = {0: "stable_or_drop", 1: "moderate_jump", 2: "high_jump"}

# ============================================================
# CONTROL MODULE
# ============================================================
PH_DELTA = 0.005
PH_LAMBDA = 0.05

DDM_WARNING_LEVEL = 2.0
DDM_DRIFT_LEVEL = 3.0

LINUCB_ALPHA = 1.0
RETRAIN_COST = 0.02
RETRAIN_COST_WEIGHT = 1.0

# ============================================================
# DSS (decision module)
# ============================================================
DSS_CATBOOST_CONFIDENCE_BUY = 0.35
DSS_LSTM_GROWTH_BUY = 0.1
DSS_CANDLE_WINDOW_SIZE = 2

# ============================================================
# FALLBACK MODE (when ML models are unavailable)
# ============================================================
# Price position thresholds (percentile 0-100)
FALLBACK_PRICE_MIN_PCT: float = float(os.getenv("FALLBACK_PRICE_MIN_PCT", "15.0"))   # <=15% of range -> BUY
FALLBACK_PRICE_MAX_PCT: float = float(os.getenv("FALLBACK_PRICE_MAX_PCT", "85.0"))   # >=85% of range -> WAIT

# Neutral zone around mean (+/- percent)
FALLBACK_NEUTRAL_ZONE_PCT: float = float(os.getenv("FALLBACK_NEUTRAL_ZONE_PCT", "10.0"))

# Minimum confidence levels
FALLBACK_CONFIDENCE_BUY_MIN: float = 0.55
FALLBACK_CONFIDENCE_WAIT_MIN: float = 0.50

# Minimum observations required for heuristic fallback
FALLBACK_MIN_OBSERVATIONS: int = int(os.getenv("FALLBACK_MIN_OBSERVATIONS", "5"))

# ============================================================
# FastAPI
# ============================================================
API_HOST = "127.0.0.1"
API_PORT = 8000
