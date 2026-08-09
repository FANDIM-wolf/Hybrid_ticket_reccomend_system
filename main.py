"""
FastAPI application of the DSS.
Endpoints:
GET  /health                          — system status
GET  /tickets                         — list of all tickets
GET  /ticket/{option}                 — single ticket card
GET  /ticket/{option}/history         — price history (flat series)
GET  /ticket/{option}/candles         — OHLC candles
GET  /recommend/{option}              — main DSS endpoint
GET  /models/status                   — active model status
GET  /control/state                   — control module state
GET  /control/logs                    — control event log
GET  /control/retrain_events          — retraining log
POST /retrain                         — forced retraining
POST /parser/run                      — launch real parser
POST /etl/run                         — run ETL on accumulated snapshot files
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from app.config import DATA_MODE
from app.db import count_observations, count_unique_tickets
from app.dss.candles import get_candles, get_price_history, get_ticket_info, list_tickets
from app.dss.decision import decide
from app.ml.training import get_active_models_info, has_active_models

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DSS Prototype — Airfare Purchase Support",
    description=(
        "Hybrid DSS for airfare purchase decisions. "
        "LSTM + CatBoost + LinUCB meta-agent with drift detectors. "
        "Fallback mode activates when ML models are unavailable."
    ),
    version="1.1.0",
)

# ============================================================
# CORS — allow requests from frontend (localhost:3000 and any origins in dev)
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# PYDANTIC RESPONSE SCHEMAS
# ============================================================
class HealthResponse(BaseModel):
    status: str
    data_mode: str
    observations: int
    unique_tickets: int
    models_active: bool
    fallback_mode: bool = Field(default=False, description="True if system is running in fallback mode (no ML models)")

class RecommendResponse(BaseModel):
    option_name: str
    current_price: float
    recommendation: str
    confidence: float
    reasoning: str
    lstm: Optional[Dict[str, Any]] = None
    catboost: Optional[Dict[str, Any]] = None
    price_history: Optional[Dict[str, float]] = None
    model_trust: Optional[Dict[str, Any]] = None
    fallback: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Present when fallback mode is active: {mode, reason, price_percentile}"
    )

class ActionResponse(BaseModel):
    status: str
    detail: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

# ============================================================
# SYSTEM
# ============================================================
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """System health check."""
    models_active = has_active_models()
    return HealthResponse(
        status="ok",
        data_mode=DATA_MODE,
        observations=count_observations(),
        unique_tickets=count_unique_tickets(),
        models_active=models_active,
        fallback_mode=not models_active,
    )

# ============================================================
# TICKETS
# ============================================================
@app.get("/tickets", tags=["Tickets"])
def tickets():
    """List of all available tickets with brief summary."""
    return {"tickets": list_tickets()}

@app.get("/ticket/{option_name}", tags=["Tickets"])
def ticket(option_name: str):
    """Detailed card of a single ticket (price, dates, statistics)."""
    info = get_ticket_info(option_name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{option_name}' not found")
    return info

@app.get("/ticket/{option_name}/history", tags=["Tickets"])
def ticket_history(
    option_name: str,
    limit: int = Query(default=500, ge=1, le=5000, description="Maximum observations"),
):
    """
    Flat time series of price for line chart.
    Fields: observation_date, time_slot, best_price, days_until_departure, jump_class.
    """
    info = get_ticket_info(option_name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{option_name}' not found")
    return {"history": get_price_history(option_name, limit=limit)}

@app.get("/ticket/{option_name}/candles", tags=["Tickets"])
def ticket_candles(
    option_name: str,
    limit: int = Query(default=60, ge=1, le=365, description="Number of days (candles)"),
):
    """
    OHLC candles by day.
    Fields: date, open, high, low, close, n_obs.
    """
    info = get_ticket_info(option_name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{option_name}' not found")
    return {"candles": get_candles(option_name, limit=limit)}

# ============================================================
# RECOMMEND (main DSS endpoint)
# ============================================================
@app.get("/recommend/{option_name}", response_model=RecommendResponse, tags=["Recommend"])
def recommend(option_name: str):
    """
    Main DSS endpoint. Returns buy/wait/neutral recommendation,
    confidence, reasoning and details of LSTM and CatBoost forecasts.

    When ML models are unavailable, returns fallback recommendation
    based on price history heuristics.
    """
    try:
        rec = decide(option_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in /recommend/{option_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal recommendation error: {e}")

    # Normalize proba keys to strings
    proba_normalized: Optional[Dict[str, float]] = None
    if rec.catboost_proba is not None:
        try:
            proba_normalized = {str(k): float(v) for k, v in rec.catboost_proba.items()}
        except Exception as e:
            logger.warning(f"Failed to normalize catboost_proba: {e}, raw={rec.catboost_proba}")
            proba_normalized = None

    # Build response
    response: Dict[str, Any] = {
        "option_name": rec.option_name,
        "current_price": rec.current_price,
        "recommendation": rec.recommendation,
        "confidence": rec.confidence,
        "reasoning": rec.reasoning,
        "lstm": {
            "pred_rub": rec.lstm_pred_rub,
            "delta_pct": rec.lstm_delta_pct,
            "version": rec.lstm_version,
        } if rec.lstm_pred_rub is not None else None,
        "catboost": {
            "class": rec.catboost_class,
            "class_name": rec.catboost_class_name,
            "proba": proba_normalized,
            "version": rec.catboost_version,
        } if rec.catboost_class is not None else None,
        "price_history": {
            "min": rec.price_history_min,
            "max": rec.price_history_max,
            "mean": rec.price_history_mean,
            "percentile": rec.price_percentile,
        },
        "model_trust": rec.model_trust,
    }

    # Add fallback info if active
    if rec.fallback_mode is not None:
        response["fallback"] = {
            "mode": rec.fallback_mode,
            "reason": rec.fallback_reason,
            "price_percentile": rec.price_percentile,
        }

    return response

# ============================================================
# MODELS STATUS
# ============================================================
@app.get("/models/status", tags=["Models"])
def models_status():
    """Status of active models: version, metrics, training date."""
    info = get_active_models_info()
    models_active = has_active_models()
    return {
        "models_active": models_active,
        "fallback_available": not models_active,
        "models": info,
    }

# ============================================================
# CONTROL MODULE
# ============================================================
@app.get("/control/state", tags=["Control"])
def control_state():
    """Current state of the control module."""
    from app.control.meta_agent import MetaAgent
    agent = MetaAgent()
    return agent.get_state_summary()

@app.get("/control/logs", tags=["Control"])
def control_logs(
    limit: int = Query(default=50, ge=1, le=500, description="Number of records"),
):
    """Last N records of the control module event log."""
    from app.db import get_conn
    with get_conn() as c:
        rows = c.execute(
            """
            SELECT id, logged_at AS created_at, event_type, ph_statistic, ddm_error_rate,
                   state_vector_json, action_chosen, expected_reward, message
            FROM control_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

@app.get("/control/retrain_events", tags=["Control"])
def retrain_events():
    """Retraining log: reason, metrics before/after, LinUCB head rewards."""
    from app.db import get_conn
    with get_conn() as c:
        rows = c.execute(
            """
            SELECT id, triggered_at AS created_at, trigger_reason, config_json,
                   lstm_mae_before, lstm_mae_after,
                   catboost_f1_before, catboost_f1_after,
                   reward_head1, reward_head2, duration_seconds
            FROM retrain_events
            ORDER BY id DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

# ============================================================
# MANUAL ACTIONS (POST)
# ============================================================
@app.post("/retrain", response_model=ActionResponse, tags=["Actions"])
def retrain():
    """Forced retraining via meta-agent."""
    try:
        from app.control.meta_agent import MetaAgent
        agent = MetaAgent()
        result = agent.tick(force_retrain=True)
        return ActionResponse(
            status="ok",
            detail="Retrain completed",
            result=result,
        )
    except Exception as e:
        logger.exception(f"Retrain failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retrain failed: {e}")

@app.post("/parser/run", response_model=ActionResponse, tags=["Actions"])
def parser_run():
    """Launch real Travelpayouts parser (DATA_MODE=real only)."""
    if DATA_MODE != "real":
        raise HTTPException(
            status_code=400,
            detail=f"Parser is only available in DATA_MODE=real (current: {DATA_MODE})",
        )
    try:
        from app.parser.travelpayouts import run_parser
        saved = run_parser()
        return ActionResponse(
            status="ok",
            detail=f"Parser completed: {len(saved)} snapshot files saved",
            result={"files": [str(p) for p in saved]},
        )
    except Exception as e:
        logger.exception(f"Parser failed: {e}")
        raise HTTPException(status_code=500, detail=f"Parser failed: {e}")

@app.post("/etl/run", response_model=ActionResponse, tags=["Actions"])
def etl_run():
    """Run ETL on accumulated snapshot files."""
    try:
        from app.etl.snapshot_loader import find_snapshot_files, load_all_snapshots
        from app.etl.feature_builder import run_etl_pipeline

        files = find_snapshot_files()
        if not files:
            raise HTTPException(status_code=404, detail="No snapshot files found")

        df_raw = load_all_snapshots(files)
        if df_raw.empty:
            raise HTTPException(status_code=422, detail="Loaded empty DataFrame from snapshots")

        n = run_etl_pipeline(df_raw, data_source=DATA_MODE, truncate=False)
        return ActionResponse(
            status="ok",
            detail=f"ETL completed: {n} rows inserted",
            result={"rows_inserted": n, "files_processed": len(files)},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"ETL failed: {e}")
        raise HTTPException(status_code=500, detail=f"ETL failed: {e}")
