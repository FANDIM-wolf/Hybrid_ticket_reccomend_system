"""
CatBoost Ensemble (15 моделей) для классификации скачков цены.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, precision_recall_fscore_support)

from app.config import (
    MODELS_DIR, JUMP_THRESHOLD_STABLE, JUMP_THRESHOLD_HIGH,
    REAL_DIRECTIONS,
)
from app.db import get_conn, register_model_version, set_active_model

logger = logging.getLogger(__name__)

CATBOOST_MODEL_DIR = MODELS_DIR / "catboost_ensemble"
CATBOOST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_ENSEMBLE = 15
SYNTHETIC_WEIGHT = 0.4
CAT_FEATURES = ["time_slot", "direction", "profile"]

FEATURES_TO_DROP = [
    "id", "jump_class", "target_class", "price_change_percent",
    "best_price", "observation_date", "observation_ts", "inserted_at",
    "ticket_key", "data_source", "option_name", "option", "date", "time",
    "run_at_utc", "airline", "flight_number",
]


def add_jump_class(df: pd.DataFrame) -> pd.DataFrame:
    a = df["price_change_percent"].abs()
    df = df.copy()
    df["jump_class"] = np.where(
        a <= JUMP_THRESHOLD_STABLE, 0,
        np.where(a <= JUMP_THRESHOLD_HIGH, 1, 2)
    )
    return df


def _clean_features(X: pd.DataFrame, cat_features: List[str]) -> pd.DataFrame:
    X = X.copy()
    for c in cat_features:
        if c in X.columns:
            X[c] = X[c].fillna("unknown").astype(str)
    for c in X.columns:
        if c in cat_features:
            continue
        s = pd.to_numeric(X[c], errors="coerce")
        s = s.fillna(50.0 if c == "rsi_7" else 0.0)
        X[c] = s
    return X


def _prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    y = df["jump_class"].astype(int).values
    drop_cols = [c for c in FEATURES_TO_DROP if c in df.columns]
    X = df.drop(columns=drop_cols).copy()
    available_cat = [c for c in CAT_FEATURES if c in X.columns]
    X = _clean_features(X, available_cat)
    return X, y, X.columns.tolist(), available_cat


def _build_class_weights(mode: str, y: np.ndarray) -> Optional[Dict[int, float]]:
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    if mode == "balanced":
        return {int(c): total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    elif mode == "boost_jump":
        w = {int(c): total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        if 2 in w: w[2] *= 4.5
        if 1 in w: w[1] *= 1.5
        return w
    return None


def _build_sample_weights(df: pd.DataFrame, indices, synthetic_weight: float) -> np.ndarray:
    is_real = df.loc[indices, "direction"].isin(REAL_DIRECTIONS).values
    return np.where(is_real, 1.0, synthetic_weight)


def _split_chronological_per_option(df: pd.DataFrame) -> Tuple[List[int], List[int], List[int]]:
    train, val, test = [], [], []
    for opt, g in df.groupby("option_name", sort=False):
        idx = g.sort_values(["observation_date", "time_encoded"]).index.tolist()
        n = len(idx)
        if n < 5:
            train += idx
            continue
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train += idx[:n_train]
        val += idx[n_train:n_train + n_val]
        test += idx[n_train + n_val:]
    return train, val, test


def _compute_metrics(y_true, y_pred) -> Dict:
    if len(y_true) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1_macro": 0.0, "accuracy": 0.0}
    return {
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _per_class_metrics(y_true, y_pred) -> Dict:
    if len(y_true) == 0:
        return {}
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {
        f"class_{c}": {"precision": float(p[i]), "recall": float(r[i]),
                       "f1": float(f[i]), "support": int(s[i])}
        for i, c in enumerate([0, 1, 2])
    }


def train_catboost_ensemble(
        df: pd.DataFrame,
        n_ensemble: int = N_ENSEMBLE,
        class_weights_mode: str = "balanced",
        synthetic_weight: float = SYNTHETIC_WEIGHT,
        base_seed: int = 1000,
        learning_rate: float = 0.03,          # <-- лучший LR (настраивается)
        test_only_real: bool = False,         # <-- тест только на реальных направлениях
) -> Dict:
    logger.info(f"CatBoost Ensemble: {len(df)} rows, {df['option_name'].nunique()} options, "
                f"n_ensemble={n_ensemble}, lr={learning_rate}, test_only_real={test_only_real}")

    df = add_jump_class(df)
    class_dist = pd.Series(df["jump_class"]).value_counts().sort_index().to_dict()
    logger.info(f"Class distribution: {class_dist}")

    X, y, feature_names, cat_features = _prepare_xy(df)
    logger.info(f"Features: {len(feature_names)} (cat: {cat_features})")

    train_idx, val_idx, test_idx = _split_chronological_per_option(df)

    # --- ФИЛЬТР ТЕСТА ТОЛЬКО НА РЕАЛЬНЫЕ НАПРАВЛЕНИЯ ---
    if test_only_real:
        test_idx = [idx for idx in test_idx if df.loc[idx, "direction"] in REAL_DIRECTIONS]
        logger.info(f"Test filtered to real directions: {len(test_idx)} samples")

    X_tr = X.loc[train_idx].reset_index(drop=True); y_tr = y[train_idx]
    X_va = X.loc[val_idx].reset_index(drop=True);   y_va = y[val_idx]
    X_te = X.loc[test_idx].reset_index(drop=True);  y_te = y[test_idx]
    logger.info(f"Split: train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")

    w_tr = _build_sample_weights(df, train_idx, synthetic_weight)
    n_real = int((w_tr == 1.0).sum())
    logger.info(f"Sample weights: real={n_real}, synth={len(w_tr) - n_real} "
                f"(synth weight={synthetic_weight})")

    cw = _build_class_weights(class_weights_mode, y_tr)

    # === Обучаем ансамбль ===
    test_probas: List[np.ndarray] = []
    val_probas: List[np.ndarray] = []
    train_probas: List[np.ndarray] = []
    individual_test_f1: List[float] = []
    model_paths: List[str] = []
    config_log = []

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for i in range(n_ensemble):
        seed = base_seed + i
        rng = np.random.default_rng(seed)
        depth_var = int(rng.choice([3, 4, 4, 4, 5]))
        l2_var = float(rng.uniform(7, 15))
        bag_temp = float(rng.uniform(0.7, 1.3))

        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=learning_rate,      # <-- используем переданный LR
            depth=depth_var,
            l2_leaf_reg=l2_var,
            border_count=64,
            bagging_temperature=bag_temp,
            random_strength=1.5,
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Macro",
            random_seed=seed,
            verbose=False,
            class_weights=cw,
        )
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            cat_features=cat_features,
            eval_set=(X_va, y_va),
            early_stopping_rounds=30,
        )

        proba_tr = model.predict_proba(X_tr)
        proba_va = model.predict_proba(X_va)
        proba_te = model.predict_proba(X_te)
        train_probas.append(proba_tr)
        val_probas.append(proba_va)
        test_probas.append(proba_te)

        ind_pred_te = np.argmax(proba_te, axis=1)
        ind_f1 = float(f1_score(y_te, ind_pred_te, average="macro", zero_division=0))
        individual_test_f1.append(ind_f1)

        mp = CATBOOST_MODEL_DIR / f"catboost_ens_{version}_m{i:02d}.cbm"
        model.save_model(str(mp))
        model_paths.append(str(mp))
        config_log.append({"seed": seed, "depth": depth_var, "l2": l2_var,
                           "bag_temp": bag_temp, "individual_f1": ind_f1})

        logger.info(f"  Model {i+1:>2}/{n_ensemble}: depth={depth_var}, l2={l2_var:.1f}, "
                    f"bag={bag_temp:.2f}, F1={ind_f1:.4f}")

    # === Усреднение и метрики ===
    avg_train = np.mean(train_probas, axis=0)
    avg_val = np.mean(val_probas, axis=0)
    avg_test = np.mean(test_probas, axis=0)
    pred_train = np.argmax(avg_train, axis=1)
    pred_val = np.argmax(avg_val, axis=1)
    pred_test = np.argmax(avg_test, axis=1)

    test_metrics = _compute_metrics(y_te, pred_test)
    ensemble_test_f1 = test_metrics["f1_macro"]
    avg_individual_f1 = float(np.mean(individual_test_f1))

    metrics = {
        "horizon": 1,
        "catboost": {
            "train": _compute_metrics(y_tr, pred_train),
            "val":   _compute_metrics(y_va, pred_val),
            "test":  test_metrics,
        },
        "test_per_class": _per_class_metrics(y_te, pred_test),
        "individual_models": {
            "f1_mean": avg_individual_f1,
            "f1_std":  float(np.std(individual_test_f1)),
            "f1_min":  float(np.min(individual_test_f1)),
            "f1_max":  float(np.max(individual_test_f1)),
        },
        "ensemble_gain_over_avg_individual": ensemble_test_f1 - avg_individual_f1,
    }

    # === Метрики только на реальных направлениях (для сверки с дипломом) ===
    real_mask = df.loc[test_idx, "direction"].isin(REAL_DIRECTIONS).values
    if real_mask.any():
        y_te_real = y_te[real_mask]
        pred_test_real = pred_test[real_mask]
        f1_real = float(f1_score(y_te_real, pred_test_real, average="macro", zero_division=0))
        metrics["test_real_directions"] = {
            "f1_macro": f1_real,
            "n_samples": int(real_mask.sum()),
            **_compute_metrics(y_te_real, pred_test_real),
            "per_class": _per_class_metrics(y_te_real, pred_test_real),
        }
        logger.info(f"F1 on REAL directions only: {f1_real:.4f} ({real_mask.sum()} samples)")

    # Baselines
    dominant = int(pd.Series(y_tr).mode()[0])
    dom_pred_te = np.full_like(y_te, dominant)
    dom_f1 = float(f1_score(y_te, dom_pred_te, average="macro", zero_division=0))
    prev_te = pd.Series(y_te).shift(1).fillna(0).astype(int).values
    pers_f1 = float(f1_score(y_te, prev_te, average="macro", zero_division=0))
    metrics["baselines"] = {
        "dominant_class_f1": dom_f1,
        "persistence_f1": pers_f1,
    }
    metrics["improvement_over_baseline"] = {
        "vs_dominant": round(ensemble_test_f1 - dom_f1, 4),
        "vs_persistence": round(ensemble_test_f1 - pers_f1, 4),
    }

    logger.info("=" * 60)
    logger.info(f"ENSEMBLE F1 macro test: {ensemble_test_f1:.4f}")
    logger.info(f"Individual mean F1: {np.mean(individual_test_f1):.4f} "
                f"± {np.std(individual_test_f1):.4f}")
    logger.info(f"Ensemble gain: {ensemble_test_f1 - np.mean(individual_test_f1):+.4f}")
    logger.info(f"vs dominant baseline: {metrics['improvement_over_baseline']['vs_dominant']:+.4f}")
    logger.info("=" * 60)

    # === Сохранение meta ===
    meta = {
        "version": version,
        "model_type": "catboost_ensemble",
        "n_ensemble": n_ensemble,
        "ensemble_paths": model_paths,
        "feature_names": feature_names,
        "cat_features": cat_features,
        "config": {
            "n_ensemble": n_ensemble,
            "class_weights_mode": class_weights_mode,
            "synthetic_weight": synthetic_weight,
            "base_seed": base_seed,
            "learning_rate": learning_rate,
            "test_only_real": test_only_real,
            "models_config": config_log,
            "jump_threshold_stable": JUMP_THRESHOLD_STABLE,
            "jump_threshold_high": JUMP_THRESHOLD_HIGH,
        },
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = CATBOOST_MODEL_DIR / f"meta_{version}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    # === Регистрация в БД ===
    # Если test_only_real=True, записываем F1 реальных как основную метрику
    # (чтобы /models/status показывал ту же цифру, что и в дипломе)
    metric_name = "test_f1_macro"
    metric_value = ensemble_test_f1
    if test_only_real and real_mask.any():
        metric_name = "test_f1_macro_real_only"
        metric_value = f1_real
        logger.info(f"Registering REAL-only F1={f1_real:.4f} as primary metric")

    register_model_version(
        model_name="catboost",
        version=version,
        model_path=str(meta_path),
        meta_path=str(meta_path),
        metric_name=metric_name,
        metric_value=metric_value,
    )
    set_active_model("catboost", version)

    logger.info(f"Saved ensemble meta: {meta_path}")
    logger.info(f"Active CatBoost: v{version}, F1={metric_value:.4f}")

    return {"version": version, "metrics": metrics, "meta_path": str(meta_path)}


# ============================================================
# INFERENCE (без изменений)
# ============================================================
_ENSEMBLE_CACHE: Dict[str, Tuple[List[CatBoostClassifier], dict]] = {}


def _load_ensemble(version: Optional[str] = None) -> Tuple[List[CatBoostClassifier], dict]:
    if version is None:
        with get_conn() as c:
            row = c.execute(
                "SELECT version, meta_path FROM models_meta "
                "WHERE model_name='catboost' AND is_active=1 LIMIT 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("No active CatBoost ensemble")
        version = row["version"]
        meta_path = row["meta_path"]
    else:
        meta_path = str(CATBOOST_MODEL_DIR / f"meta_{version}.json")

    if version in _ENSEMBLE_CACHE:
        return _ENSEMBLE_CACHE[version]

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    models = []
    for mp in meta["ensemble_paths"]:
        m = CatBoostClassifier()
        m.load_model(mp)
        models.append(m)

    _ENSEMBLE_CACHE[version] = (models, meta)
    return models, meta


def predict_catboost(option_name: str) -> Optional[Dict]:
    try:
        models, meta = _load_ensemble()
    except Exception as e:
        logger.warning(f"Ensemble load failed: {e}")
        return None

    feature_names = meta["feature_names"]
    cat_features = meta["cat_features"]

    with get_conn() as c:
        row = c.execute(
            """
            SELECT * FROM observations
            WHERE option_name = ?
            ORDER BY observation_date DESC, time_encoded DESC
            LIMIT 1
            """, (option_name,)
        ).fetchone()
    if row is None:
        logger.warning(f"No observations for {option_name}")
        return None

    last = pd.DataFrame([dict(row)])
    last = _clean_features(last, cat_features)

    for col in feature_names:
        if col not in last.columns:
            last[col] = "unknown" if col in cat_features else 0.0
    X_pred = last[feature_names].copy()

    probas = []
    for m in models:
        proba = m.predict_proba(X_pred)[0]
        probas.append(proba)
    avg_proba = np.mean(probas, axis=0)
    predicted_class = int(np.argmax(avg_proba))

    return {
        "model_version": meta["version"],
        "ensemble_size": len(models),
        "predicted_class": predicted_class,
        "proba": {
            "0": float(avg_proba[0]),
            "1": float(avg_proba[1]),
            "2": float(avg_proba[2]),
        },
    }