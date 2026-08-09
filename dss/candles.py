"""
Модуль для получения данных о билетах: список, карточка, история цен, свечи.
Используется эндпоинтами /tickets, /ticket/{option}, /ticket/{option}/history, /ticket/{option}/candles
"""
from __future__ import annotations
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from app.db import get_conn

def list_tickets() -> List[Dict]:
    """Возвращает краткую сводку по всем доступным билетам."""
    with get_conn() as c:
        df = pd.read_sql_query(
            """
            SELECT option_name, direction, profile,
                   COUNT(*) as n_obs,
                   MIN(best_price) as min_price,
                   MAX(best_price) as max_price,
                   AVG(best_price) as mean_price,
                   MIN(observation_date) as first_date,
                   MAX(observation_date) as last_date
            FROM observations
            GROUP BY option_name, direction, profile
            ORDER BY option_name
            """,
            c,
        )
        if df.empty:
            return []
        return df.to_dict(orient="records")

def get_ticket_info(option_name: str) -> Optional[Dict]:
    """
    Подробная карточка одного билета.
    Возвращает текущую цену, статистику, даты вылета/прилёта и детали рейсов.
    """
    with get_conn() as c:
        df = pd.read_sql_query(
            """
            SELECT best_price, observation_date, days_until_departure, flight_duration_minutes,
                   direction,
                   outbound_flight, outbound_aircraft,
                   outbound_depart_time, outbound_arrival_time,
                   outbound_duration_min, outbound_depart_airport, outbound_arrival_airport,
                   outbound_date,
                   return_flight, return_aircraft,
                   return_depart_time, return_arrival_time,
                   return_duration_min, return_depart_airport, return_arrival_airport,
                   return_date
            FROM observations
            WHERE option_name = ?
            ORDER BY observation_date DESC, time_encoded DESC
            LIMIT 100
            """,
            c,
            params=(option_name,),
        )
        if df.empty:
            return None

        latest = df.iloc[0]
        current_price = float(latest["best_price"])

        obs_date = pd.to_datetime(latest["observation_date"]).date()
        days_until = int(latest["days_until_departure"]) if pd.notna(latest["days_until_departure"]) else 0
        duration = int(latest["flight_duration_minutes"]) if pd.notna(latest["flight_duration_minutes"]) else 0

        departure_date = obs_date + timedelta(days=days_until)
        arrival_date = departure_date + timedelta(minutes=duration)

        def _str(val):
            return str(val) if pd.notna(val) else None

        def _int(val):
            return int(val) if pd.notna(val) else None

        return {
            "option_name": option_name,
            "direction": _str(latest.get("direction")),
            "current_price": current_price,
            "min_price": float(df["best_price"].min()),
            "max_price": float(df["best_price"].max()),
            "mean_price": float(df["best_price"].mean()),
            "departure_date": departure_date.isoformat(),
            "arrival_date": arrival_date.isoformat(),
            "flight_duration_minutes": duration,
            "last_observation_date": _str(latest["observation_date"]),
            "n_obs": len(df),
            # Детали исходящего рейса
            "outbound": {
                "flight":           _str(latest.get("outbound_flight")),
                "aircraft":         _str(latest.get("outbound_aircraft")),
                "depart_time":      _str(latest.get("outbound_depart_time")),
                "arrival_time":     _str(latest.get("outbound_arrival_time")),
                "duration_min":     _int(latest.get("outbound_duration_min")),
                "depart_airport":   _str(latest.get("outbound_depart_airport")),
                "arrival_airport":  _str(latest.get("outbound_arrival_airport")),
                "date":             _str(latest.get("outbound_date")),
            },
            # Детали обратного рейса
            "return": {
                "flight":           _str(latest.get("return_flight")),
                "aircraft":         _str(latest.get("return_aircraft")),
                "depart_time":      _str(latest.get("return_depart_time")),
                "arrival_time":     _str(latest.get("return_arrival_time")),
                "duration_min":     _int(latest.get("return_duration_min")),
                "depart_airport":   _str(latest.get("return_depart_airport")),
                "arrival_airport":  _str(latest.get("return_arrival_airport")),
                "date":             _str(latest.get("return_date")),
            },
        }

def get_candles(option_name: str, limit: int = 60) -> List[Dict]:
    """
    Возвращает последние N свечей по дням для указанного билета.
    Формат одной свечи:
    { "date": "2026-04-12", "open": 22150, "high": 22340, "low": 21980, "close": 22200, "n_obs": 2}
    """
    with get_conn() as c:
        df = pd.read_sql_query(
            """
            SELECT observation_date, time_encoded, best_price
            FROM observations
            WHERE option_name = ?
            ORDER BY observation_date, time_encoded
            """,
            c,
            params=(option_name,),
        )
        if df.empty:
            return []

        candles = []
        for date, group in df.groupby("observation_date"):
            group = group.sort_values("time_encoded")
            prices = group["best_price"].astype(float).values
            candles.append({
                "date": str(date),
                "open": float(prices[0]),
                "high": float(prices.max()),
                "low": float(prices.min()),
                "close": float(prices[-1]),
                "n_obs": int(len(prices)),
            })
        return candles[-limit:]

def get_price_history(option_name: str, limit: int = 500) -> List[Dict]:
    """
    Возвращает последние N наблюдений цены (плоский временной ряд для графика).
    Используется эндпоинтом /ticket/{option}/history.
    """
    with get_conn() as c:
        df = pd.read_sql_query(
            """
            SELECT observation_date, time_slot, time_encoded, best_price,
                   days_until_departure, jump_class
            FROM observations
            WHERE option_name = ?
            ORDER BY observation_date, time_encoded
            """,
            c,
            params=(option_name,),
        )
        if df.empty:
            return []
        records = df.to_dict(orient="records")
        return records[-limit:]