#!/usr/bin/env python3
"""OMEN OPUS — Read-only OMEN event analytics, forecasting, research, and PPTrader.

This is a web application only. It deliberately omits the live Chromium Tracker.
Import Tracker/Predictor/Chainlink/settlement CSV files through the UI.

Run:
    pip install -r requirements.txt
    streamlit run omen_opus_webapp.py
"""

import hashlib
import json
import math
import re
import sqlite3
import statistics
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

APP_NAME = "OMEN OPUS"
DB_PATH = Path("omen_opus.sqlite3")
EXPORT_DIR = Path("output")
EXPORT_DIR.mkdir(exist_ok=True)
UTC = timezone.utc

SOURCE_FILES = {
    "twap": "omen_twap_observations.csv",
    "exact_calls": "omen_exact_calls.csv",
    "changes": "omen_yes_no_changes.csv",
    "multipliers": "omen_call_multipliers.csv",
    "predictor_calls": "omen_predictor_calls.csv",
    "predictor_diagnostics": "omen_predictor_diagnostics.csv",
    "chainlink": "chainlink_twap_observations.csv",
    "settlements": "omen_settlements.csv",
}

SOURCE_LABELS = {
    "twap": "Tracker TWAP observations",
    "exact_calls": "Official Tracker calls",
    "changes": "YES/NO multiplier changes",
    "multipliers": "Five-slot multiplier captures",
    "predictor_calls": "Predictor calls",
    "predictor_diagnostics": "Predictor diagnostics",
    "chainlink": "Chainlink TWAP-60",
    "settlements": "Settlement outcomes",
}


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_now() -> str:
    return utc_now().isoformat()


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    try:
        text = str(value).replace("$", "").replace(",", "").strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return default
        return float(text)
    except Exception:
        return default


def safe_text(value: Any, default: str = "") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return str(value).strip()


def normalize_direction(value: Any) -> str:
    raw = safe_text(value).upper().replace("-", "_").replace(" ", "_")
    if raw in {"UP", "LONG", "YES", "YES_UP", "TARGET_UP", "BUY"}:
        return "UP"
    if raw in {"DOWN", "SHORT", "NO", "YES_DOWN", "TARGET_DOWN", "SELL"}:
        return "DOWN"
    return "NEUTRAL"


def parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def choose_column(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    normalized = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    return None


def row_value(row: pd.Series, columns: Iterable[str], candidates: List[str], default=None):
    col = choose_column(columns, candidates)
    return row[col] if col and col in row else default


def stable_hash(*parts: Any) -> str:
    text = "|".join(safe_text(p) for p in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cycle_bounds(ts: datetime) -> Tuple[datetime, datetime, str, str, int]:
    """Return 5-min cycle start/end, id, M1-M5 slot, and seconds into cycle."""
    ts = ts.astimezone(UTC)
    minute = (ts.minute // 5) * 5
    start = ts.replace(minute=minute, second=0, microsecond=0)
    end = start + timedelta(minutes=5)
    elapsed = int((ts - start).total_seconds())
    slot = f"M{min(5, elapsed // 60 + 1)}"
    cycle_id = start.strftime("%Y%m%dT%H%MZ")
    return start, end, cycle_id, slot, elapsed


class Store:
    """SQLite persistence layer. Dashboard and Control tables remain separated."""

    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self.init()

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init(self):
        with self.connection() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS import_batches (
                batch_id TEXT PRIMARY KEY,
                imported_at TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                rows_seen INTEGER NOT NULL,
                rows_inserted INTEGER NOT NULL,
                rows_duplicate INTEGER NOT NULL,
                rows_rejected INTEGER NOT NULL,
                notes TEXT
            );
            CREATE TABLE IF NOT EXISTS raw_records (
                record_hash TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                source_name TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                timestamp_utc TEXT,
                event_id TEXT,
                question_id TEXT,
                raw_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_raw_source ON raw_records(source_type, event_id, timestamp_utc);

            CREATE TABLE IF NOT EXISTS dashboard_events (
                event_id TEXT PRIMARY KEY,
                question_id TEXT,
                market_title TEXT,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS dashboard_observations (
                observation_hash TEXT PRIMARY KEY,
                timestamp_utc TEXT NOT NULL,
                event_id TEXT NOT NULL,
                question_id TEXT,
                market_title TEXT,
                cycle_id TEXT NOT NULL,
                cycle_start_utc TEXT NOT NULL,
                cycle_end_utc TEXT NOT NULL,
                frame_slot TEXT NOT NULL,
                frame_second INTEGER NOT NULL,
                btc_twap REAL,
                target_price REAL,
                target_status TEXT NOT NULL,
                up_multiplier REAL,
                down_multiplier REAL,
                implied_probability_up REAL,
                implied_probability_down REAL,
                multiplier_edge REAL,
                volume REAL,
                source_name TEXT NOT NULL,
                imported_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_obs_event_time ON dashboard_observations(event_id, timestamp_utc);
            CREATE INDEX IF NOT EXISTS idx_obs_cycle ON dashboard_observations(event_id, cycle_id);

            CREATE TABLE IF NOT EXISTS dashboard_calls (
                call_hash TEXT PRIMARY KEY,
                timestamp_utc TEXT NOT NULL,
                event_id TEXT NOT NULL,
                question_id TEXT,
                cycle_id TEXT NOT NULL,
                frame_slot TEXT NOT NULL,
                signal_id TEXT NOT NULL,
                direction TEXT NOT NULL,
                call_source TEXT NOT NULL,
                confidence REAL,
                target_price REAL,
                twap_at_call REAL,
                target_distance REAL,
                up_multiplier_at_call REAL,
                down_multiplier_at_call REAL,
                implied_probability_up REAL,
                implied_probability_down REAL,
                price_freshness_seconds REAL,
                multiplier_freshness_seconds REAL,
                is_reversal INTEGER NOT NULL DEFAULT 0,
                prior_signal_id TEXT,
                explanation TEXT,
                raw_json TEXT,
                imported_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_calls_event_cycle ON dashboard_calls(event_id, cycle_id, timestamp_utc);

            CREATE TABLE IF NOT EXISTS dashboard_predictions (
                prediction_hash TEXT PRIMARY KEY,
                timestamp_utc TEXT NOT NULL,
                event_id TEXT NOT NULL,
                question_id TEXT,
                cycle_id TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL,
                predicted_price REAL,
                current_price REAL,
                expected_return REAL,
                model_version TEXT,
                features_json TEXT,
                source_name TEXT NOT NULL,
                imported_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_predictions_event_cycle ON dashboard_predictions(event_id, cycle_id, timestamp_utc);

            CREATE TABLE IF NOT EXISTS dashboard_forecasts (
                forecast_hash TEXT PRIMARY KEY,
                timestamp_utc TEXT NOT NULL,
                event_id TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                forecast_twap_60 REAL NOT NULL,
                target_price REAL,
                target_cross_probability REAL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                current_twap REAL,
                methodology TEXT NOT NULL,
                features_json TEXT,
                settled_twap REAL,
                forecast_error REAL,
                scored_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_forecasts_event_cycle ON dashboard_forecasts(event_id, cycle_id, timestamp_utc);

            CREATE TABLE IF NOT EXISTS dashboard_settlements (
                settlement_hash TEXT PRIMARY KEY,
                timestamp_utc TEXT NOT NULL,
                event_id TEXT NOT NULL,
                question_id TEXT,
                cycle_id TEXT NOT NULL,
                target_price REAL,
                final_twap_60 REAL NOT NULL,
                resolved_direction TEXT NOT NULL,
                source_name TEXT NOT NULL,
                imported_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_settlements_event_cycle ON dashboard_settlements(event_id, cycle_id);

            CREATE TABLE IF NOT EXISTS control_overrides (
                override_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                event_id TEXT NOT NULL,
                cycle_id TEXT,
                override_type TEXT NOT NULL,
                override_value REAL,
                note TEXT,
                active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS control_notes (
                note_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                event_id TEXT,
                cycle_id TEXT,
                note TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pptrader_trades (
                trade_id TEXT PRIMARY KEY,
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                event_id TEXT NOT NULL,
                cycle_id TEXT,
                strategy_name TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                entry_multiplier REAL,
                position_value REAL NOT NULL,
                commission REAL NOT NULL,
                pnl REAL,
                pnl_pct REAL,
                status TEXT NOT NULL,
                signal_source TEXT NOT NULL,
                metadata_json TEXT
            );
            CREATE TABLE IF NOT EXISTS pptrader_backtests (
                backtest_id TEXT PRIMARY KEY,
                run_at TEXT NOT NULL,
                event_id TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                initial_capital REAL NOT NULL,
                final_capital REAL NOT NULL,
                total_return REAL NOT NULL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                total_trades INTEGER NOT NULL,
                profit_factor REAL,
                parameters_json TEXT,
                equity_json TEXT
            );
            """)

    def table(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with self.connection() as c:
            return pd.read_sql_query(sql, c, params=params)

    def scalar(self, sql: str, params: tuple = (), default=None):
        with self.connection() as c:
            row = c.execute(sql, params).fetchone()
            return row[0] if row else default

    def write_import_batch(self, **kwargs):
        with self.connection() as c:
            c.execute("""
                INSERT OR REPLACE INTO import_batches VALUES(?,?,?,?,?,?,?,?,?,?)
            """, (kwargs["batch_id"], kwargs["imported_at"], kwargs["source_type"], kwargs["source_name"],
                  kwargs["source_sha256"], kwargs["rows_seen"], kwargs["rows_inserted"],
                  kwargs["rows_duplicate"], kwargs["rows_rejected"], kwargs.get("notes", "")))


STORE = Store()


def upsert_event(conn, event_id: str, question_id: str, title: str, timestamp: str):
    conn.execute("""
        INSERT INTO dashboard_events(event_id,question_id,market_title,first_seen,last_seen,active)
        VALUES(?,?,?,?,?,1)
        ON CONFLICT(event_id) DO UPDATE SET
          question_id=COALESCE(NULLIF(excluded.question_id,''),dashboard_events.question_id),
          market_title=COALESCE(NULLIF(excluded.market_title,''),dashboard_events.market_title),
          last_seen=excluded.last_seen, active=1
    """, (event_id, question_id, title, timestamp, timestamp))


def latest_observation_before(conn, event_id: str, ts: datetime) -> Optional[sqlite3.Row]:
    return conn.execute("""
        SELECT * FROM dashboard_observations
        WHERE event_id=? AND timestamp_utc<=?
        ORDER BY timestamp_utc DESC LIMIT 1
    """, (event_id, ts.isoformat())).fetchone()


def latest_multiplier_before(conn, event_id: str, ts: datetime) -> Optional[sqlite3.Row]:
    return conn.execute("""
        SELECT * FROM dashboard_observations
        WHERE event_id=? AND timestamp_utc<=? AND up_multiplier IS NOT NULL
        ORDER BY timestamp_utc DESC LIMIT 1
    """, (event_id, ts.isoformat())).fetchone()


def implied_probs(up: Optional[float], down: Optional[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if up is None or down is None or up <= 0 or down <= 0:
        return None, None, None
    total = up + down
    p_up = up / total
    p_down = down / total
    return p_up, p_down, p_up - p_down


def import_csv(source_type: str, source_name: str, payload: bytes) -> Dict[str, Any]:
    """Import one CSV safely. Raw row fingerprints prevent duplicate ingestion."""
    if source_type not in SOURCE_FILES:
        raise ValueError("Unsupported source type")
    file_hash = hashlib.sha256(payload).hexdigest()
    batch_id = stable_hash(source_type, source_name, file_hash)
    try:
        df = pd.read_csv(pd.io.common.BytesIO(payload))
    except Exception as exc:
        return {"error": f"CSV could not be read: {exc}"}
    df.columns = [str(c).strip() for c in df.columns]
    inserted = duplicate = rejected = 0
    imported_at = iso_now()
    with STORE.connection() as conn:
        for _, row in df.iterrows():
            raw = {col: (None if pd.isna(row[col]) else row[col]) for col in df.columns}
            ts = parse_timestamp(row_value(row, df.columns, ["timestamp", "timestamp_utc", "time", "observed_at", "created_at"]))
            event_id = safe_text(row_value(row, df.columns, ["event_id", "market_id", "event", "market"], ""))
            question_id = safe_text(row_value(row, df.columns, ["question_id", "questionid"], ""))
            raw_hash = stable_hash(source_type, source_name, json.dumps(raw, sort_keys=True, default=str))
            already = conn.execute("SELECT 1 FROM raw_records WHERE record_hash=?", (raw_hash,)).fetchone()
            if already:
                duplicate += 1
                continue
            conn.execute("""
                INSERT INTO raw_records(record_hash,source_type,source_name,imported_at,timestamp_utc,event_id,question_id,raw_json)
                VALUES(?,?,?,?,?,?,?,?)
            """, (raw_hash, source_type, source_name, imported_at, ts.isoformat() if ts else None,
                  event_id, question_id, json.dumps(raw, default=str)))
            try:
                if source_type in {"twap", "multipliers", "changes", "chainlink"}:
                    ingest_observation(conn, source_type, raw, ts, event_id, question_id, source_name, raw_hash)
                elif source_type == "exact_calls":
                    ingest_call(conn, raw, ts, event_id, question_id, source_name, raw_hash, "OFFICIAL_TRACKER")
                elif source_type == "predictor_calls":
                    ingest_prediction(conn, raw, ts, event_id, question_id, source_name, raw_hash)
                elif source_type == "settlements":
                    ingest_settlement(conn, raw, ts, event_id, question_id, source_name, raw_hash)
                # diagnostics remain in raw_records intentionally; they are exportable/auditable
                inserted += 1
            except Exception:
                rejected += 1
        conn.commit()
    STORE.write_import_batch(batch_id=batch_id, imported_at=imported_at, source_type=source_type,
                             source_name=source_name, source_sha256=file_hash, rows_seen=len(df),
                             rows_inserted=inserted, rows_duplicate=duplicate, rows_rejected=rejected,
                             notes="Raw rows are retained even when a derived record is rejected.")
    rebuild_fusion_calls()
    score_settled_forecasts()
    export_all()
    return {"rows_seen": len(df), "inserted": inserted, "duplicates": duplicate, "rejected": rejected}


def ingest_observation(conn, source_type, raw, ts, event_id, question_id, source_name, raw_hash):
    if not ts or not event_id:
        raise ValueError("Observation requires timestamp and event_id")
    columns = raw.keys()
    title = safe_text(next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"markettitle","question","title","eventtitle"}), ""))
    twap = safe_float(next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"twapprice","btctwap","twap","price","pricevalue","finaltwap60"}), None))
    target = safe_float(next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"targetprice","target","settlementtarget","targetline"}), None))
    up = safe_float(next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"upmultiplier","yesmultiplier","multiplierup","slot5","outcomeaprice"}), None))
    down = safe_float(next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"downmultiplier","nomultiplier","multiplierdown","outcomebprice"}), None))
    if down is None and up is not None and 0 < up < 1:
        down = 1 - up
    volume = safe_float(next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"volume","volumeusd","liquidity"}), None))
    target_status = "VALID" if target is not None else "MISSING"
    start, end, cycle_id, slot, frame_second = cycle_bounds(ts)
    p_up, p_down, edge = implied_probs(up, down)
    upsert_event(conn, event_id, question_id, title, ts.isoformat())
    conn.execute("""
        INSERT OR IGNORE INTO dashboard_observations(
          observation_hash,timestamp_utc,event_id,question_id,market_title,cycle_id,cycle_start_utc,cycle_end_utc,
          frame_slot,frame_second,btc_twap,target_price,target_status,up_multiplier,down_multiplier,
          implied_probability_up,implied_probability_down,multiplier_edge,volume,source_name,imported_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (raw_hash, ts.isoformat(), event_id, question_id, title, cycle_id, start.isoformat(), end.isoformat(),
          slot, frame_second, twap, target, target_status, up, down, p_up, p_down, edge, volume, source_name, iso_now()))


def ingest_call(conn, raw, ts, event_id, question_id, source_name, raw_hash, call_source):
    if not ts or not event_id:
        raise ValueError("Call requires timestamp and event_id")
    columns = raw.keys()
    direction = normalize_direction(next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"calltype","direction","signal","side"}), ""))
    if direction == "NEUTRAL":
        raise ValueError("Neutral call ignored")
    confidence = safe_float(next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"confidence","signalstrength","strength"}), None))
    target = safe_float(next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"targetprice","target","settlementtarget"}), None))
    start, end, cycle_id, slot, _ = cycle_bounds(ts)
    obs = latest_observation_before(conn, event_id, ts)
    mult = latest_multiplier_before(conn, event_id, ts)
    twap = obs["btc_twap"] if obs else None
    if target is None and obs:
        target = obs["target_price"]
    up = mult["up_multiplier"] if mult else None
    down = mult["down_multiplier"] if mult else None
    p_up, p_down, _ = implied_probs(up, down)
    price_age = (ts - parse_timestamp(obs["timestamp_utc"])).total_seconds() if obs else None
    mult_age = (ts - parse_timestamp(mult["timestamp_utc"])).total_seconds() if mult else None
    target_distance = (twap - target) if twap is not None and target is not None else None
    prior = conn.execute("""
        SELECT signal_id,direction FROM dashboard_calls
        WHERE event_id=? AND cycle_id=? ORDER BY timestamp_utc DESC LIMIT 1
    """, (event_id, cycle_id)).fetchone()
    reversal = int(bool(prior and prior["direction"] != direction))
    signal_id = stable_hash(event_id, cycle_id, call_source, ts.isoformat(), direction)
    explanation = f"{call_source}: {direction}; frozen at imported call timestamp."
    conn.execute("""
        INSERT OR IGNORE INTO dashboard_calls(
          call_hash,timestamp_utc,event_id,question_id,cycle_id,frame_slot,signal_id,direction,call_source,
          confidence,target_price,twap_at_call,target_distance,up_multiplier_at_call,down_multiplier_at_call,
          implied_probability_up,implied_probability_down,price_freshness_seconds,multiplier_freshness_seconds,
          is_reversal,prior_signal_id,explanation,raw_json,imported_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (raw_hash, ts.isoformat(), event_id, question_id, cycle_id, slot, signal_id, direction, call_source,
          confidence, target, twap, target_distance, up, down, p_up, p_down, price_age, mult_age,
          reversal, prior["signal_id"] if prior else None, explanation, json.dumps(raw, default=str), iso_now()))


def ingest_prediction(conn, raw, ts, event_id, question_id, source_name, raw_hash):
    if not ts or not event_id:
        raise ValueError("Prediction requires timestamp and event_id")
    columns = raw.keys()
    signal_raw = next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"signal","direction","prediction"}), "NEUTRAL")
    direction = normalize_direction(signal_raw)
    confidence = safe_float(next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"confidence","signalstrength"}), None), 0.0)
    predicted = safe_float(next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"predictedprice","forecastprice","predictionprice"}), None))
    current = safe_float(next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"currentprice","twapprice","price"}), None))
    expected = safe_float(next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"expectedreturn","return"}), None))
    model = safe_text(next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"modelversion","model","strategy"}), "unknown"))
    features = safe_text(next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"featuresjson","features"}), "{}"))
    start, end, cycle_id, slot, _ = cycle_bounds(ts)
    upsert_event(conn, event_id, question_id, "", ts.isoformat())
    conn.execute("""
        INSERT OR IGNORE INTO dashboard_predictions(
          prediction_hash,timestamp_utc,event_id,question_id,cycle_id,signal,confidence,predicted_price,current_price,
          expected_return,model_version,features_json,source_name,imported_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (raw_hash, ts.isoformat(), event_id, question_id, cycle_id, direction, confidence, predicted, current,
          expected, model, features, source_name, iso_now()))
    if direction != "NEUTRAL":
        ingest_call(conn, {"direction": direction, "confidence": confidence, "target_price": None}, ts,
                    event_id, question_id, source_name, stable_hash(raw_hash, "predictor-call"), "PREDICTOR")


def ingest_settlement(conn, raw, ts, event_id, question_id, source_name, raw_hash):
    if not event_id:
        raise ValueError("Settlement requires event_id")
    ts = ts or utc_now()
    columns = raw.keys()
    final_twap = safe_float(next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"finaltwap60","finaltwap","settlementtwap","twapprice","twap"}), None))
    if final_twap is None:
        raise ValueError("Settlement requires final TWAP")
    target = safe_float(next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"targetprice","target","settlementtarget"}), None))
    start, end, cycle_id, _, _ = cycle_bounds(ts)
    if safe_text(raw.get("cycle_id", "")):
        cycle_id = safe_text(raw["cycle_id"])
    direction = normalize_direction(next((raw[c] for c in columns if re.sub(r"[^a-z0-9]", "", c.lower()) in {"resolveddirection","direction","result","outcome"}), ""))
    if direction == "NEUTRAL" and target is not None:
        direction = "UP" if final_twap >= target else "DOWN"
    upsert_event(conn, event_id, question_id, "", ts.isoformat())
    conn.execute("""
        INSERT OR IGNORE INTO dashboard_settlements(
          settlement_hash,timestamp_utc,event_id,question_id,cycle_id,target_price,final_twap_60,resolved_direction,source_name,imported_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?)
    """, (raw_hash, ts.isoformat(), event_id, question_id, cycle_id, target, final_twap, direction, source_name, iso_now()))


def rebuild_fusion_calls():
    """Create composite calls without altering imported official/predictor records."""
    with STORE.connection() as conn:
        events = [r[0] for r in conn.execute("SELECT DISTINCT event_id FROM dashboard_observations").fetchall()]
        for event_id in events:
            rows = conn.execute("""
                SELECT * FROM dashboard_observations WHERE event_id=?
                ORDER BY timestamp_utc DESC LIMIT 1
            """, (event_id,)).fetchall()
            if not rows:
                continue
            obs = rows[0]
            ts = parse_timestamp(obs["timestamp_utc"])
            if not ts:
                continue
            cycle_id = obs["cycle_id"]
            votes: List[Tuple[str, float, str]] = []
            # latest official/predictor call in current cycle
            calls = conn.execute("""
                SELECT direction,confidence,call_source FROM dashboard_calls
                WHERE event_id=? AND cycle_id=? AND call_source IN ('OFFICIAL_TRACKER','PREDICTOR')
                ORDER BY timestamp_utc DESC
            """, (event_id, cycle_id)).fetchall()
            seen = set()
            for call in calls:
                if call["call_source"] in seen:
                    continue
                seen.add(call["call_source"])
                votes.append((call["direction"], float(call["confidence"] or 0.5), call["call_source"]))
            # multiplier vote
            if obs["implied_probability_up"] is not None:
                p = float(obs["implied_probability_up"])
                if p > 0.505:
                    votes.append(("UP", min(0.95, 0.5 + abs(p - 0.5) * 2), "MULTIPLIER"))
                elif p < 0.495:
                    votes.append(("DOWN", min(0.95, 0.5 + abs(p - 0.5) * 2), "MULTIPLIER"))
            if not votes:
                continue
            up_score = sum(weight for d, weight, _ in votes if d == "UP")
            down_score = sum(weight for d, weight, _ in votes if d == "DOWN")
            total = up_score + down_score
            if total <= 0:
                continue
            direction = "UP" if up_score > down_score else "DOWN" if down_score > up_score else "NEUTRAL"
            confidence = max(up_score, down_score) / total
            call_hash = stable_hash("fusion", event_id, cycle_id, obs["timestamp_utc"], direction, round(confidence, 5))
            raw = {"direction": direction, "confidence": confidence, "target_price": obs["target_price"]}
            # Avoid producing an identical current signal repeatedly
            exists = conn.execute("SELECT 1 FROM dashboard_calls WHERE call_hash=?", (call_hash,)).fetchone()
            if not exists and direction != "NEUTRAL":
                ingest_call(conn, raw, ts, event_id, obs["question_id"], "fusion-engine", call_hash, "FUSION")
                conn.execute("UPDATE dashboard_calls SET explanation=? WHERE call_hash=?", (
                    "Composite agreement: " + ", ".join(f"{src}={d}" for d, _, src in votes), call_hash))


def create_terminal_forecast(event_id: str) -> Optional[Dict[str, Any]]:
    """Terminal TWAP estimate from recent trend, target proximity, and multiplier probability."""
    obs = STORE.table("""
        SELECT * FROM dashboard_observations WHERE event_id=? ORDER BY timestamp_utc DESC LIMIT 180
    """, (event_id,))
    if obs.empty:
        return None
    obs = obs.sort_values("timestamp_utc")
    latest = obs.iloc[-1]
    current = safe_float(latest["btc_twap"])
    if current is None:
        return None
    target = safe_float(latest["target_price"])
    recent = obs["btc_twap"].dropna().tail(20).astype(float)
    slope = float((recent.iloc[-1] - recent.iloc[0]) / max(len(recent)-1, 1)) if len(recent) > 1 else 0.0
    velocity = slope * 12  # approximate 60s projection for 5s observations; still valid as a heuristic with sparse source data
    p_up = safe_float(latest["implied_probability_up"], 0.5)
    market_bias = (p_up - 0.5) * max(float(recent.std()) if len(recent) > 2 else 0.01, 0.005)
    forecast = current + velocity + market_bias
    volatility = float(recent.pct_change().dropna().std()) if len(recent) > 3 else 0.01
    confidence = float(min(0.90, max(0.35, 0.75 - volatility * 20 + abs(p_up - 0.5) * 0.4)))
    if target is None:
        direction = "UP" if forecast >= current else "DOWN"
        cross_prob = p_up
    else:
        scale = max(abs(float(recent.std())) if len(recent) > 2 else 0.01, 0.003)
        z = (forecast - target) / scale
        cross_prob = float(1 / (1 + math.exp(-z)))
        direction = "UP" if cross_prob >= 0.5 else "DOWN"
    ts = utc_now()
    _, _, cycle_id, _, _ = cycle_bounds(ts)
    forecast_hash = stable_hash("forecast", event_id, cycle_id, ts.replace(second=0, microsecond=0).isoformat())
    features = {"current_twap": current, "trend_per_observation": slope, "market_probability_up": p_up,
                "recent_observations": len(recent), "volatility": volatility, "target": target}
    with STORE.connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO dashboard_forecasts(
              forecast_hash,timestamp_utc,event_id,cycle_id,forecast_twap_60,target_price,target_cross_probability,
              direction,confidence,current_twap,methodology,features_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        """, (forecast_hash, ts.isoformat(), event_id, cycle_id, forecast, target, cross_prob, direction,
              confidence, current, "trend + multiplier-implied probability + target relation", json.dumps(features)))
    return {"forecast": forecast, "target": target, "cross_probability": cross_prob, "direction": direction,
            "confidence": confidence, "features": features}


def score_settled_forecasts():
    with STORE.connection() as conn:
        rows = conn.execute("""
            SELECT f.forecast_hash,s.final_twap_60,s.timestamp_utc
            FROM dashboard_forecasts f JOIN dashboard_settlements s
              ON f.event_id=s.event_id AND f.cycle_id=s.cycle_id
            WHERE f.scored_at IS NULL
        """).fetchall()
        for row in rows:
            forecast = conn.execute("SELECT forecast_twap_60 FROM dashboard_forecasts WHERE forecast_hash=?", (row["forecast_hash"],)).fetchone()
            error = abs(float(forecast[0]) - float(row["final_twap_60"]))
            conn.execute("UPDATE dashboard_forecasts SET settled_twap=?,forecast_error=?,scored_at=? WHERE forecast_hash=?",
                         (row["final_twap_60"], error, iso_now(), row["forecast_hash"]))


def source_health() -> pd.DataFrame:
    rows = []
    now = utc_now()
    for source_type, label in SOURCE_LABELS.items():
        raw = STORE.table("""
            SELECT timestamp_utc,imported_at,source_name FROM raw_records WHERE source_type=?
            ORDER BY imported_at DESC LIMIT 1
        """, (source_type,))
        count = STORE.scalar("SELECT COUNT(*) FROM raw_records WHERE source_type=?", (source_type,), 0)
        if raw.empty:
            status, age, last = "MISSING", None, None
        else:
            last = parse_timestamp(raw.iloc[0]["timestamp_utc"]) or parse_timestamp(raw.iloc[0]["imported_at"])
            age = (now-last).total_seconds() if last else None
            if source_type == "chainlink" and count == 0:
                status = "HEADER_ONLY"
            elif age is None:
                status = "UNKNOWN"
            elif age <= 10:
                status = "FRESH"
            elif age <= 60:
                status = "AGING"
            else:
                status = "STALE"
        rows.append({"Source": label, "Source Key": source_type, "Rows": count, "Status": status,
                     "Age Seconds": round(age, 1) if age is not None else None,
                     "Last Timestamp": last.isoformat() if last else None})
    return pd.DataFrame(rows)


def export_all():
    exports = {
        "omen_opus_observations.csv": "SELECT * FROM dashboard_observations ORDER BY timestamp_utc",
        "omen_opus_calls.csv": "SELECT * FROM dashboard_calls ORDER BY timestamp_utc",
        "omen_opus_predictions.csv": "SELECT * FROM dashboard_predictions ORDER BY timestamp_utc",
        "omen_opus_forecasts.csv": "SELECT * FROM dashboard_forecasts ORDER BY timestamp_utc",
        "omen_opus_settlements.csv": "SELECT * FROM dashboard_settlements ORDER BY timestamp_utc",
        "omen_opus_pptrader_trades.csv": "SELECT * FROM pptrader_trades ORDER BY opened_at",
        "omen_opus_import_history.csv": "SELECT * FROM import_batches ORDER BY imported_at DESC",
    }
    for filename, query in exports.items():
        STORE.table(query).to_csv(EXPORT_DIR / filename, index=False)


def current_state(event_id: str) -> Dict[str, Any]:
    obs = STORE.table("SELECT * FROM dashboard_observations WHERE event_id=? ORDER BY timestamp_utc DESC LIMIT 1", (event_id,))
    if obs.empty:
        return {}
    row = obs.iloc[0].to_dict()
    now = utc_now()
    ts = parse_timestamp(row["timestamp_utc"])
    row["price_freshness_seconds"] = (now-ts).total_seconds() if ts else None
    mult = row["price_freshness_seconds"] if row.get("up_multiplier") is not None else None
    row["multiplier_freshness_seconds"] = mult
    calls = STORE.table("SELECT * FROM dashboard_calls WHERE event_id=? AND cycle_id=? ORDER BY timestamp_utc DESC", (event_id,row["cycle_id"]))
    row["latest_call"] = calls.iloc[0].to_dict() if not calls.empty else None
    return row


def event_ids() -> List[str]:
    df = STORE.table("SELECT event_id FROM dashboard_events WHERE active=1 ORDER BY last_seen DESC")
    return df["event_id"].tolist() if not df.empty else []


def event_title(event_id: str) -> str:
    df = STORE.table("SELECT market_title FROM dashboard_events WHERE event_id=?", (event_id,))
    if df.empty or not safe_text(df.iloc[0]["market_title"]):
        return event_id
    return f"{event_id} — {df.iloc[0]['market_title']}"


def add_override(event_id: str, cycle_id: str, value: Optional[float], note: str):
    override_id = stable_hash("override", event_id, cycle_id, iso_now(), value, note)
    with STORE.connection() as conn:
        conn.execute("""
          INSERT INTO control_overrides(override_id,created_at,event_id,cycle_id,override_type,override_value,note,active)
          VALUES(?,?,?,?,?,?,?,1)
        """, (override_id, iso_now(), event_id, cycle_id, "TARGET_OVERRIDE", value, note))


def pptrader_open(event_id: str, strategy: str, side: str, amount: float, signal_source: str):
    state = current_state(event_id)
    price = safe_float(state.get("btc_twap"))
    if price is None or price <= 0:
        return "No valid TWAP price exists for this event."
    trade_id = stable_hash("paper", event_id, strategy, side, iso_now())[:24]
    commission = amount * 0.001
    with STORE.connection() as conn:
        conn.execute("""
          INSERT INTO pptrader_trades(trade_id,opened_at,event_id,cycle_id,strategy_name,side,entry_price,entry_multiplier,
            position_value,commission,status,signal_source,metadata_json)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (trade_id, iso_now(), event_id, state.get("cycle_id"), strategy, side, price,
              state.get("up_multiplier") if side == "UP" else state.get("down_multiplier"), amount, commission,
              "OPEN", signal_source, json.dumps({"mode":"paper_only"})))
    return f"Opened paper {side} position: {trade_id}"


def pptrader_close(event_id: str):
    state = current_state(event_id)
    exit_price = safe_float(state.get("btc_twap"))
    if exit_price is None:
        return "No valid price exists for closing."
    with STORE.connection() as conn:
        trades = conn.execute("SELECT * FROM pptrader_trades WHERE event_id=? AND status='OPEN'", (event_id,)).fetchall()
        for t in trades:
            entry = float(t["entry_price"])
            value = float(t["position_value"])
            commission = float(t["commission"])
            pnl = ((exit_price-entry)/entry*value if t["side"] == "UP" else (entry-exit_price)/entry*value) - commission
            pnl_pct = pnl/value*100 if value else 0
            conn.execute("UPDATE pptrader_trades SET closed_at=?,exit_price=?,pnl=?,pnl_pct=?,status='CLOSED' WHERE trade_id=?",
                         (iso_now(), exit_price, pnl, pnl_pct, t["trade_id"]))
    export_all()
    return f"Closed {len(trades)} paper position(s)."


def run_backtest(event_id: str, strategy: str, capital: float, position_pct: float, commission_pct: float) -> Tuple[Dict[str, Any], pd.DataFrame]:
    obs = STORE.table("SELECT * FROM dashboard_observations WHERE event_id=? ORDER BY timestamp_utc", (event_id,))
    calls = STORE.table("SELECT * FROM dashboard_calls WHERE event_id=? ORDER BY timestamp_utc", (event_id,))
    if len(obs) < 3:
        return {"error":"Need at least three observations."}, pd.DataFrame()
    obs["timestamp_utc"] = pd.to_datetime(obs["timestamp_utc"], utc=True)
    calls["timestamp_utc"] = pd.to_datetime(calls["timestamp_utc"], utc=True) if not calls.empty else pd.Series(dtype="datetime64[ns, UTC]")
    equity = capital
    curve = [{"timestamp":obs.iloc[0]["timestamp_utc"],"equity":equity}]
    pnls = []
    for i in range(1, len(obs)):
        prev, cur = obs.iloc[i-1], obs.iloc[i]
        recent_calls = calls[calls["timestamp_utc"] <= cur["timestamp_utc"]]
        direction = None
        if strategy == "Strategy A — First Call":
            cycle_calls = recent_calls[recent_calls["cycle_id"] == cur["cycle_id"]]
            if not cycle_calls.empty:
                direction = cycle_calls.iloc[0]["direction"]
        elif strategy == "Strategy B — Swap":
            if not recent_calls.empty:
                direction = recent_calls.iloc[-1]["direction"]
        elif strategy == "Strategy C — Predictor Only":
            pred = recent_calls[recent_calls["call_source"] == "PREDICTOR"]
            if not pred.empty: direction = pred.iloc[-1]["direction"]
        elif strategy == "Strategy D — Dashboard Only":
            fusion = recent_calls[recent_calls["call_source"] == "FUSION"]
            if not fusion.empty: direction = fusion.iloc[-1]["direction"]
        elif strategy == "Strategy E — Hybrid":
            current_cycle = recent_calls[recent_calls["cycle_id"] == cur["cycle_id"]]
            if not current_cycle.empty:
                vote = current_cycle.groupby("direction")["confidence"].mean()
                if not vote.empty: direction = vote.idxmax()
        else:
            delta = safe_float(cur["btc_twap"],0)-safe_float(prev["btc_twap"],0)
            direction = "UP" if delta >= 0 else "DOWN"
        p0, p1 = safe_float(prev["btc_twap"]), safe_float(cur["btc_twap"])
        if direction in {"UP","DOWN"} and p0 and p1 and p0 > 0:
            value = equity * position_pct
            ret = (p1-p0)/p0 if direction == "UP" else (p0-p1)/p0
            pnl = value*ret - value*commission_pct
            equity += pnl
            pnls.append(pnl)
        curve.append({"timestamp":cur["timestamp_utc"],"equity":equity})
    arr = np.array([x["equity"] for x in curve])
    peaks = np.maximum.accumulate(arr)
    max_dd = float(np.min((arr-peaks)/peaks)*100) if len(arr) else 0
    rets = np.diff(arr)/np.maximum(arr[:-1],1)
    stats = {"strategy":strategy,"initial_capital":capital,"final_capital":equity,"total_return":(equity/capital-1)*100,
             "total_trades":len(pnls),"win_rate":(sum(x>0 for x in pnls)/len(pnls)*100) if pnls else 0,
             "sharpe_ratio":float(np.mean(rets)/(np.std(rets)+1e-9)*np.sqrt(252)) if len(rets)>1 else 0,
             "max_drawdown":max_dd,"profit_factor":float(sum(x for x in pnls if x>0)/abs(sum(x for x in pnls if x<0))) if any(x<0 for x in pnls) else 0}
    backtest_id = stable_hash("backtest",event_id,strategy,iso_now())
    with STORE.connection() as conn:
        conn.execute("""
          INSERT INTO pptrader_backtests(backtest_id,run_at,event_id,strategy_name,initial_capital,final_capital,total_return,
          sharpe_ratio,max_drawdown,win_rate,total_trades,profit_factor,parameters_json,equity_json)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (backtest_id,iso_now(),event_id,strategy,capital,equity,stats["total_return"],stats["sharpe_ratio"],
              max_dd,stats["win_rate"],stats["total_trades"],stats["profit_factor"],json.dumps({"position_pct":position_pct,"commission_pct":commission_pct}),
              json.dumps([{**x,"timestamp":x["timestamp"].isoformat()} for x in curve])))
    return stats, pd.DataFrame(curve)


def research(event_id: str) -> Dict[str, Any]:
    settlements = STORE.table("SELECT * FROM dashboard_settlements WHERE event_id=?", (event_id,))
    calls = STORE.table("SELECT * FROM dashboard_calls WHERE event_id=? ORDER BY timestamp_utc", (event_id,))
    if settlements.empty or calls.empty:
        return {"available":False,"message":"Import settlement results plus call history to score call accuracy."}
    records = []
    for _, s in settlements.iterrows():
        cycle_calls = calls[calls["cycle_id"] == s["cycle_id"]]
        if cycle_calls.empty: continue
        first = cycle_calls.iloc[0]
        final = cycle_calls.iloc[-1]
        outcome = s["resolved_direction"]
        records.append({"cycle_id":s["cycle_id"],"outcome":outcome,"first":first["direction"],"final":final["direction"],
                        "first_correct":int(first["direction"]==outcome),"final_correct":int(final["direction"]==outcome),
                        "reversals":int(cycle_calls["is_reversal"].sum()),"final_slot":final["frame_slot"],
                        "final_confidence":safe_float(final["confidence"],0)})
    if not records:
        return {"available":False,"message":"No cycles contain both settlement and call data."}
    df = pd.DataFrame(records)
    n=len(df)
    first_acc=df.first_correct.mean(); final_acc=df.final_correct.mean()
    def ci(p):
        se=math.sqrt(p*(1-p)/max(n,1)); return (max(0,p-1.96*se),min(1,p+1.96*se))
    return {"available":True,"cycles":n,"first_accuracy":first_acc,"first_ci":ci(first_acc),"final_accuracy":final_acc,
            "final_ci":ci(final_acc),"avg_reversals":df.reversals.mean(),"late_final_share":(df.final_slot.isin(["M5"])).mean(),"records":df}


# ------------------------- Streamlit User Interface -------------------------
st.set_page_config(page_title="OMEN OPUS", page_icon="◉", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {padding-top:1.2rem; max-width:1450px;}
[data-testid='stMetric'] {background:#111a26; border:1px solid #263544; padding:0.7rem; border-radius:10px;}
.small-muted {color:#93a4b6; font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("◉ OMEN OPUS")
    st.caption("Read-only monitoring, forecasting, research")
    page = st.radio("Application", ["Live Dashboard","Control Center","Import & Health","Research","PPTrader","Exports & Setup"])
    ids = event_ids()
    selected = st.selectbox("Event", ids, format_func=event_title) if ids else None
    st.divider()
    st.success("READ-ONLY MODE")
    st.caption("No browser tracker, orders, accounts, wallets, or live trading actions.")

if page == "Import & Health":
    st.title("Import & Feed Health")
    st.write("Upload CSV outputs from your external Tracker, Predictor, Chainlink integration, or settlement process. Importing is restart-safe and duplicate-safe.")
    left, right = st.columns([1,1])
    with left:
        source_type = st.selectbox("Data source", list(SOURCE_FILES.keys()), format_func=lambda x: f"{SOURCE_LABELS[x]} — {SOURCE_FILES[x]}")
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded and st.button("Import CSV", type="primary"):
            result = import_csv(source_type, uploaded.name, uploaded.getvalue())
            if "error" in result: st.error(result["error"])
            else: st.success(f"Imported: {result['inserted']} derived rows; {result['duplicates']} duplicate rows skipped; {result['rejected']} rejected rows.")
    with right:
        st.subheader("Expected schemas")
        st.code("timestamp,event_id,question_id,twap_price,target_price,up_multiplier,down_multiplier,volume", language="text")
        st.caption("Column aliases are accepted. At minimum, observations require timestamp and event_id. Settlements additionally need final_twap_60.")
        st.info("Chainlink fallback: if Chainlink is missing/stale/header-only, the dashboard explicitly uses Tracker TWAP data as the reference fallback.")
    st.subheader("Feed Health")
    health = source_health()
    st.dataframe(health, use_container_width=True, hide_index=True)
    st.subheader("Import History")
    st.dataframe(STORE.table("SELECT * FROM import_batches ORDER BY imported_at DESC LIMIT 100"), use_container_width=True, hide_index=True)

elif not selected:
    st.title("OMEN OPUS")
    st.info("Start on **Import & Health** and upload at least one Tracker TWAP observations CSV. Once an event is imported, it will appear in the sidebar.")
    st.code("timestamp,event_id,question_id,twap_price,target_price,up_multiplier,down_multiplier,volume", language="text")

elif page == "Live Dashboard":
    st.title("Live OMEN Dashboard")
    st.caption(event_title(selected))
    state = current_state(selected)
    if not state:
        st.warning("No observations available for this event.")
        st.stop()
    price_age = state.get("price_freshness_seconds")
    mult_age = state.get("multiplier_freshness_seconds")
    freshness = "FRESH" if price_age is not None and price_age <= 10 else "AGING" if price_age is not None and price_age <= 60 else "STALE"
    a,b,c,d,e,f = st.columns(6)
    a.metric("BTC / TWAP", f"{safe_float(state.get('btc_twap'),0):.6f}")
    b.metric("Settlement Target", f"{safe_float(state.get('target_price'),0):.6f}" if state.get("target_price") is not None else "Missing")
    c.metric("UP Multiplier", f"{safe_float(state.get('up_multiplier'),0):.6f}" if state.get("up_multiplier") is not None else "—")
    d.metric("DOWN Multiplier", f"{safe_float(state.get('down_multiplier'),0):.6f}" if state.get("down_multiplier") is not None else "—")
    e.metric("Implied UP", f"{safe_float(state.get('implied_probability_up'),0):.1%}" if state.get("implied_probability_up") is not None else "—")
    f.metric("Price Feed", freshness, f"{price_age:.1f}s old" if price_age is not None else "unknown age")
    st.caption(f"Cycle {state['cycle_id']} • {state['frame_slot']} • Target status: {state['target_status']} • Multiplier age: {mult_age:.1f}s" if mult_age is not None else f"Cycle {state['cycle_id']} • {state['frame_slot']} • Target status: {state['target_status']}")
    col1,col2 = st.columns([3,1])
    with col1:
        obs = STORE.table("SELECT * FROM dashboard_observations WHERE event_id=? ORDER BY timestamp_utc DESC LIMIT 1000", (selected,))
        obs = obs.sort_values("timestamp_utc")
        fig = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.10,subplot_titles=("TWAP / Target / Multipliers","Implied Probability and Volume"))
        fig.add_trace(go.Scatter(x=obs.timestamp_utc,y=obs.btc_twap,name="TWAP",line=dict(color="#00d4ff",width=3)),row=1,col=1)
        if obs.target_price.notna().any(): fig.add_trace(go.Scatter(x=obs.timestamp_utc,y=obs.target_price,name="Target",line=dict(color="#ffb000",dash="dash")),row=1,col=1)
        if obs.up_multiplier.notna().any(): fig.add_trace(go.Scatter(x=obs.timestamp_utc,y=obs.up_multiplier,name="UP mult",line=dict(color="#00d17a")),row=1,col=1)
        if obs.down_multiplier.notna().any(): fig.add_trace(go.Scatter(x=obs.timestamp_utc,y=obs.down_multiplier,name="DOWN mult",line=dict(color="#ff4b5c")),row=1,col=1)
        if obs.implied_probability_up.notna().any(): fig.add_trace(go.Scatter(x=obs.timestamp_utc,y=obs.implied_probability_up,name="P(UP)",line=dict(color="#ad82ff")),row=2,col=1)
        if obs.volume.notna().any(): fig.add_trace(go.Bar(x=obs.timestamp_utc,y=obs.volume,name="Volume",marker_color="#395c80",opacity=0.55),row=2,col=1)
        fig.update_layout(template="plotly_dark",height=680,hovermode="x unified",legend=dict(orientation="h"))
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        st.subheader("Current Signal")
        call=state.get("latest_call")
        if call:
            st.metric(call["call_source"],call["direction"],f"confidence {safe_float(call['confidence'],0):.1%}")
            st.caption(call.get("explanation", ""))
            st.write(f"Reversal: {'Yes' if call['is_reversal'] else 'No'}")
        else: st.info("No call recorded for the current cycle.")
        if st.button("Create Terminal-TWAP Forecast",type="primary"):
            result=create_terminal_forecast(selected)
            if result: st.success(f"Forecast: {result['direction']} at {result['confidence']:.0%} confidence")
            else: st.error("Need a valid TWAP observation.")
            st.rerun()
    st.subheader("M1–M5 Signal Capture Timeline")
    calls=STORE.table("SELECT * FROM dashboard_calls WHERE event_id=? AND cycle_id=? ORDER BY timestamp_utc",(selected,state["cycle_id"]))
    if calls.empty: st.info("No captured calls in this cycle.")
    else:
        st.dataframe(calls[["timestamp_utc","frame_slot","direction","call_source","confidence","is_reversal","up_multiplier_at_call","down_multiplier_at_call","price_freshness_seconds","multiplier_freshness_seconds"]],use_container_width=True,hide_index=True)

elif page == "Control Center":
    st.title("Control Center")
    st.caption("Independent read-only supervision, forecasting, and audited companion overrides.")
    state=current_state(selected)
    c1,c2,c3=st.columns(3)
    c1.metric("Tracker TWAP Health", source_health().query("`Source Key`=='twap'").iloc[0]["Status"])
    c2.metric("Current Cycle", state.get("cycle_id","—"))
    c3.metric("Frame",state.get("frame_slot","—"))
    st.subheader("Settlement-TWAP Forecasts")
    if st.button("Generate Forecast",type="primary"):
        outcome=create_terminal_forecast(selected)
        if outcome: st.success("Forecast stored independently from Dashboard state.")
        st.rerun()
    forecasts=STORE.table("SELECT * FROM dashboard_forecasts WHERE event_id=? ORDER BY timestamp_utc DESC LIMIT 100",(selected,))
    if not forecasts.empty:
        latest=forecasts.iloc[0]
        a,b,c,d=st.columns(4)
        a.metric("Forecast TWAP-60",f"{latest.forecast_twap_60:.6f}")
        b.metric("Direction",latest.direction)
        c.metric("Target-cross probability",f"{latest.target_cross_probability:.1%}" if pd.notna(latest.target_cross_probability) else "—")
        d.metric("Confidence",f"{latest.confidence:.1%}")
        st.dataframe(forecasts[["timestamp_utc","cycle_id","forecast_twap_60","target_price","target_cross_probability","direction","confidence","settled_twap","forecast_error"]],use_container_width=True,hide_index=True)
    else: st.info("No forecasts yet.")
    st.subheader("Companion Target Override")
    st.caption("Overrides are stored only in the Control database/table. They never alter Tracker observations or Dashboard target history.")
    with st.form("override"):
        value=st.number_input("Companion target value",value=float(state.get("target_price") or 0.0),format="%.8f")
        note=st.text_input("Audit note",placeholder="Why this companion override is being recorded")
        submitted=st.form_submit_button("Record Read-only Override")
    if submitted:
        add_override(selected,state.get("cycle_id"),value,note)
        st.success("Audited companion override recorded.")
    overrides=STORE.table("SELECT * FROM control_overrides WHERE event_id=? ORDER BY created_at DESC",(selected,))
    st.dataframe(overrides,use_container_width=True,hide_index=True)

elif page == "Research":
    st.title("Historical Research")
    r=research(selected)
    if not r["available"]:
        st.info(r["message"])
    else:
        a,b,c,d,e=st.columns(5)
        a.metric("Scored Cycles",r["cycles"])
        b.metric("First-call Accuracy",f"{r['first_accuracy']:.1%}",f"CI {r['first_ci'][0]:.1%}–{r['first_ci'][1]:.1%}")
        c.metric("Final-call Accuracy",f"{r['final_accuracy']:.1%}",f"CI {r['final_ci'][0]:.1%}–{r['final_ci'][1]:.1%}")
        d.metric("Avg Reversals",f"{r['avg_reversals']:.2f}")
        e.metric("Final Calls in M5",f"{r['late_final_share']:.1%}")
        st.warning("Interpretation guardrail: high final-call accuracy may be late and not practically actionable. Review final-slot concentration and frozen payout multipliers before treating accuracy as tradable performance.")
        records=r["records"]
        st.subheader("Cycle Results")
        st.dataframe(records,use_container_width=True,hide_index=True)
        calls=STORE.table("SELECT * FROM dashboard_calls WHERE event_id=?",(selected,))
        if not calls.empty:
            slots=calls.groupby(["frame_slot","direction"]).size().reset_index(name="count")
            fig=go.Figure()
            for direction,color in [("UP","#00d17a"),("DOWN","#ff4b5c")]:
                x=slots[slots.direction==direction]
                fig.add_trace(go.Bar(x=x.frame_slot,y=x["count"],name=direction,marker_color=color))
            fig.update_layout(template="plotly_dark",barmode="group",title="Signal Persistence by M1–M5",height=400)
            st.plotly_chart(fig,use_container_width=True)
        settled=STORE.table("SELECT forecast_error FROM dashboard_forecasts WHERE event_id=? AND forecast_error IS NOT NULL",(selected,))
        if not settled.empty: st.metric("Mean Absolute Forecast Error",f"{settled.forecast_error.mean():.6f}")

elif page == "PPTrader":
    st.title("PPTrader — Separate Paper Trading and Backtesting")
    st.warning("PPTrader is isolated from OMEN data collection. It creates simulated paper records only; it never sends orders or accesses accounts.")
    tabs=st.tabs(["Paper Positions","Backtesting","Strategy Comparison"])
    with tabs[0]:
        a,b,c=st.columns(3)
        strategy=a.selectbox("Strategy",["Strategy A — First Call","Strategy B — Swap","Strategy C — Predictor Only","Strategy D — Dashboard Only","Strategy E — Hybrid","Strategy F — Custom Rule Engine"])
        side=b.selectbox("Paper Side",["UP","DOWN"])
        amount=c.number_input("Position Value",min_value=10.0,value=1000.0,step=50.0)
        x,y=st.columns(2)
        if x.button("Open Paper Position",type="primary"):
            st.success(pptrader_open(selected,strategy,side,amount,"manual_paper"))
            st.rerun()
        if y.button("Close Event Positions"):
            st.success(pptrader_close(selected))
            st.rerun()
        trades=STORE.table("SELECT * FROM pptrader_trades WHERE event_id=? ORDER BY opened_at DESC",(selected,))
        if not trades.empty:
            m1,m2,m3,m4=st.columns(4)
            closed=trades[trades.status=="CLOSED"]
            m1.metric("Open",int((trades.status=="OPEN").sum()))
            m2.metric("Closed",len(closed))
            m3.metric("Closed P&L",f"${closed.pnl.fillna(0).sum():,.2f}")
            m4.metric("Win Rate",f"{(closed.pnl>0).mean():.1%}" if len(closed) else "—")
            st.dataframe(trades,use_container_width=True,hide_index=True)
    with tabs[1]:
        a,b,c=st.columns(3)
        strategy=a.selectbox("Backtest strategy",["Strategy A — First Call","Strategy B — Swap","Strategy C — Predictor Only","Strategy D — Dashboard Only","Strategy E — Hybrid","Strategy F — Custom Rule Engine"],key="backtest_strategy")
        capital=b.number_input("Initial capital",min_value=100.0,value=10000.0,step=500.0)
        pct=c.slider("Position percent",0.01,0.5,0.1,0.01)
        commission=st.number_input("Round-trip commission fraction",min_value=0.0,value=0.001,step=0.0005,format="%.4f")
        if st.button("Run Backtest",type="primary"):
            stats,curve=run_backtest(selected,strategy,capital,pct,commission)
            st.session_state["bt_stats"],st.session_state["bt_curve"]=stats,curve
        if "bt_stats" in st.session_state:
            stats=st.session_state["bt_stats"]; curve=st.session_state["bt_curve"]
            if "error" in stats: st.error(stats["error"])
            else:
                a,b,c,d,e=st.columns(5)
                a.metric("Return",f"{stats['total_return']:.2f}%")
                b.metric("Sharpe",f"{stats['sharpe_ratio']:.2f}")
                c.metric("Max DD",f"{stats['max_drawdown']:.2f}%")
                d.metric("Win Rate",f"{stats['win_rate']:.1f}%")
                e.metric("Trades",stats['total_trades'])
                fig=go.Figure(go.Scatter(x=curve.timestamp,y=curve.equity,name="Equity",line=dict(color="#00d4ff",width=3),fill="tozeroy"))
                fig.update_layout(template="plotly_dark",height=380,title="Paper Backtest Equity")
                st.plotly_chart(fig,use_container_width=True)
    with tabs[2]:
        results=STORE.table("SELECT * FROM pptrader_backtests WHERE event_id=? ORDER BY run_at DESC",(selected,))
        st.dataframe(results,use_container_width=True,hide_index=True)

else:
    st.title("Exports & Setup")
    st.subheader("Required deployment files")
    st.code("omen_opus_webapp.py\nrequirements.txt\nREADME.md",language="text")
    st.subheader("CSV Exports")
    export_all()
    for path in sorted(EXPORT_DIR.glob("*.csv")):
        st.download_button(f"Download {path.name}",path.read_bytes(),file_name=path.name,mime="text/csv")
    st.subheader("Database Status")
    st.write(f"SQLite database: `{DB_PATH}`")
    st.write(f"Raw input rows: {STORE.scalar('SELECT COUNT(*) FROM raw_records')}")
    st.write(f"Derived observations: {STORE.scalar('SELECT COUNT(*) FROM dashboard_observations')}")
    st.write(f"Signal captures: {STORE.scalar('SELECT COUNT(*) FROM dashboard_calls')}")
    st.write(f"Forecasts: {STORE.scalar('SELECT COUNT(*) FROM dashboard_forecasts')}")
    st.write(f"Paper trades: {STORE.scalar('SELECT COUNT(*) FROM pptrader_trades')}")
    st.info("For persistent cloud history beyond Streamlit Cloud's ephemeral filesystem, replace the SQLite Store adapter with Supabase/Postgres. The schema separation is already reflected in Dashboard, Control, and PPTrader tables.")

st.divider()
st.caption("OMEN OPUS • Strictly read-only analytics, forecasting, research, and paper trading • No live tracker/browser component included • No trade execution")
