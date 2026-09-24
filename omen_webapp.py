#!/usr/bin/env python3
"""OMEN Platform + PPTrader — single-file Streamlit web app.
Read-only analytics and simulated paper-trading/backtesting demo.
Run locally: pip install -r requirements.txt && streamlit run omen_webapp.py
"""

import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(".")
OUTPUT_DIR = APP_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
DB_PATH = APP_DIR / "omen_platform.sqlite3"


def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        event_id TEXT NOT NULL,
        question TEXT NOT NULL,
        twap_price REAL NOT NULL,
        yes_multiplier REAL NOT NULL,
        no_multiplier REAL NOT NULL,
        direction TEXT NOT NULL,
        volume REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_obs_event_time ON observations(event_id, timestamp);
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        event_id TEXT NOT NULL,
        signal TEXT NOT NULL,
        confidence REAL NOT NULL,
        predicted_price REAL NOT NULL,
        current_price REAL NOT NULL,
        source TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS paper_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_id TEXT UNIQUE NOT NULL,
        timestamp TEXT NOT NULL,
        event_id TEXT NOT NULL,
        strategy_name TEXT NOT NULL,
        side TEXT NOT NULL,
        entry_price REAL NOT NULL,
        exit_price REAL,
        position_size REAL NOT NULL,
        status TEXT NOT NULL,
        pnl REAL,
        pnl_pct REAL,
        signal_source TEXT NOT NULL
    );
    """)
    conn.commit()
    conn.close()


EVENTS = {
    "btc_1h": "Will BTC be above $65,000 in 1 hour?",
    "btc_4h": "Will BTC be above $67,000 in 4 hours?",
    "btc_24h": "Will BTC be above $70,000 in 24 hours?",
    "eth_1h": "Will ETH be above $3,500 in 1 hour?",
}


def seed_data(rows_per_event=140):
    conn = db()
    existing = conn.execute("SELECT COUNT(*) AS n FROM observations").fetchone()["n"]
    if existing > 0:
        conn.close()
        return
    rng = np.random.default_rng(42)
    now = datetime.now(timezone.utc)
    for event_index, (event_id, question) in enumerate(EVENTS.items()):
        price = 0.38 + event_index * 0.09
        for i in range(rows_per_event):
            price = float(np.clip(price + rng.normal(0, 0.012), 0.03, 0.97))
            yes = float(np.clip(price + rng.normal(0, 0.007), 0.01, 0.99))
            no = 1 - yes
            direction = "UP" if i == 0 or yes >= price else "DOWN"
            timestamp = (now - timedelta(minutes=(rows_per_event - i) * 3)).isoformat()
            conn.execute("""
                INSERT INTO observations(timestamp,event_id,question,twap_price,yes_multiplier,no_multiplier,direction,volume)
                VALUES(?,?,?,?,?,?,?,?)
            """, (timestamp, event_id, question, price, yes, no, direction, float(rng.uniform(1000, 250000))))
    conn.commit()
    conn.close()
    export_csvs()


def export_csvs():
    conn = db()
    obs = pd.read_sql_query("SELECT * FROM observations ORDER BY timestamp", conn)
    preds = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp", conn)
    trades = pd.read_sql_query("SELECT * FROM paper_trades ORDER BY timestamp", conn)
    conn.close()
    obs.to_csv(OUTPUT_DIR / "omen_twap_observations.csv", index=False)
    obs[["timestamp","event_id","yes_multiplier","no_multiplier","direction"]].to_csv(OUTPUT_DIR / "omen_call_multipliers.csv", index=False)
    obs[["timestamp","event_id","direction"]].to_csv(OUTPUT_DIR / "omen_yes_no_changes.csv", index=False)
    preds.to_csv(OUTPUT_DIR / "omen_predictor_calls.csv", index=False)
    pd.DataFrame(columns=["timestamp","event_id","call_type","target_price","confidence"]).to_csv(OUTPUT_DIR / "omen_exact_calls.csv", index=False)
    pd.DataFrame(columns=["timestamp","event_id","feature_name","feature_value"]).to_csv(OUTPUT_DIR / "omen_predictor_diagnostics.csv", index=False)
    trades.to_csv(OUTPUT_DIR / "pptrader_paper_trades.csv", index=False)


def add_live_observation(event_id):
    conn = db()
    last = conn.execute("SELECT * FROM observations WHERE event_id=? ORDER BY timestamp DESC LIMIT 1", (event_id,)).fetchone()
    rng = np.random.default_rng()
    current = float(last["twap_price"]) if last else 0.5
    price = float(np.clip(current + rng.normal(0, 0.011), 0.02, 0.98))
    yes = float(np.clip(price + rng.normal(0, 0.005), 0.01, 0.99))
    direction = "UP" if yes >= float(last["yes_multiplier"]) else "DOWN"
    conn.execute("""
        INSERT INTO observations(timestamp,event_id,question,twap_price,yes_multiplier,no_multiplier,direction,volume)
        VALUES(?,?,?,?,?,?,?,?)
    """, (datetime.now(timezone.utc).isoformat(), event_id, EVENTS[event_id], price, yes, 1-yes, direction, float(rng.uniform(1000,250000))))
    conn.commit()
    conn.close()
    export_csvs()


def generate_prediction(event_id):
    conn = db()
    df = pd.read_sql_query("SELECT * FROM observations WHERE event_id=? ORDER BY timestamp", conn, params=(event_id,))
    if len(df) < 5:
        conn.close()
        return None
    current = float(df.iloc[-1].twap_price)
    momentum = float(df.twap_price.tail(5).iloc[-1] - df.twap_price.tail(5).iloc[0])
    mult_bias = float((df.yes_multiplier.tail(5) - df.no_multiplier.tail(5)).mean())
    score = momentum * 8 + mult_bias * 0.35
    confidence = float(min(0.94, max(0.51, 0.50 + abs(score))))
    signal = "LONG" if score > 0.015 else "SHORT" if score < -0.015 else "NEUTRAL"
    predicted = float(np.clip(current + score * 0.05, 0.01, 0.99))
    conn.execute("""
        INSERT INTO predictions(timestamp,event_id,signal,confidence,predicted_price,current_price,source)
        VALUES(?,?,?,?,?,?,?)
    """, (datetime.now(timezone.utc).isoformat(), event_id, signal, confidence, predicted, current, "fusion_ensemble"))
    conn.commit()
    conn.close()
    export_csvs()
    return {"signal": signal, "confidence": confidence, "predicted": predicted, "current": current}


def latest_data(event_id):
    conn = db()
    obs = pd.read_sql_query("SELECT * FROM observations WHERE event_id=? ORDER BY timestamp", conn, params=(event_id,))
    pred = pd.read_sql_query("SELECT * FROM predictions WHERE event_id=? ORDER BY timestamp DESC LIMIT 1", conn, params=(event_id,))
    trades = pd.read_sql_query("SELECT * FROM paper_trades WHERE event_id=? ORDER BY timestamp DESC", conn, params=(event_id,))
    conn.close()
    return obs, pred, trades


def open_paper_trade(event_id, strategy, side, capital, size_pct):
    conn = db()
    row = conn.execute("SELECT * FROM observations WHERE event_id=? ORDER BY timestamp DESC LIMIT 1", (event_id,)).fetchone()
    if not row:
        conn.close()
        return None
    price = float(row["twap_price"])
    size = (capital * size_pct) / price
    trade_id = f"PP-{event_id}-{int(time.time())}"
    conn.execute("""
        INSERT INTO paper_trades(trade_id,timestamp,event_id,strategy_name,side,entry_price,position_size,status,signal_source)
        VALUES(?,?,?,?,?,?,?,?,?)
    """, (trade_id, datetime.now(timezone.utc).isoformat(), event_id, strategy, side, price, size, "OPEN", "manual_paper"))
    conn.commit()
    conn.close()
    export_csvs()
    return trade_id


def close_all_positions(event_id):
    conn = db()
    row = conn.execute("SELECT twap_price FROM observations WHERE event_id=? ORDER BY timestamp DESC LIMIT 1", (event_id,)).fetchone()
    if not row:
        conn.close()
        return
    exit_price = float(row["twap_price"])
    open_trades = conn.execute("SELECT * FROM paper_trades WHERE event_id=? AND status='OPEN'", (event_id,)).fetchall()
    for trade in open_trades:
        entry = float(trade["entry_price"])
        size = float(trade["position_size"])
        pnl = (exit_price - entry) * size if trade["side"] == "LONG" else (entry - exit_price) * size
        pnl_pct = pnl / (entry * size) * 100 if entry * size else 0
        conn.execute("UPDATE paper_trades SET status='CLOSED',exit_price=?,pnl=?,pnl_pct=? WHERE trade_id=?", (exit_price,pnl,pnl_pct,trade["trade_id"]))
    conn.commit()
    conn.close()
    export_csvs()


def run_backtest(event_id, strategy, initial_capital, size_pct, commission):
    conn = db()
    df = pd.read_sql_query("SELECT * FROM observations WHERE event_id=? ORDER BY timestamp", conn, params=(event_id,))
    conn.close()
    if len(df) < 20:
        return None, pd.DataFrame()
    capital = initial_capital
    equity = [capital]
    trades = []
    pos = None
    for i in range(5, len(df)-1):
        current = float(df.iloc[i].twap_price)
        prior = float(df.iloc[i-5].twap_price)
        change = current - prior
        signal = "LONG" if change > 0.008 else "SHORT" if change < -0.008 else "NEUTRAL"
        future = float(df.iloc[i+1].twap_price)
        if signal != "NEUTRAL":
            value = capital * size_pct
            units = value / current
            pnl = ((future-current)*units if signal == "LONG" else (current-future)*units) - value*commission*2
            capital += pnl
            trades.append(pnl)
        equity.append(capital)
    returns = np.diff(equity) / np.maximum(np.array(equity[:-1]), 1)
    peak = np.maximum.accumulate(equity)
    drawdown = (np.array(equity)-peak)/peak
    stats = {
        "strategy": strategy,
        "initial_capital": initial_capital,
        "final_capital": capital,
        "total_return": (capital/initial_capital-1)*100,
        "total_trades": len(trades),
        "win_rate": (sum(1 for x in trades if x > 0)/len(trades)*100) if trades else 0,
        "sharpe": float(np.mean(returns)/(np.std(returns)+1e-9)*np.sqrt(252)) if len(returns)>1 else 0,
        "max_drawdown": float(drawdown.min()*100),
        "profit_factor": float(sum(x for x in trades if x>0)/abs(sum(x for x in trades if x<0))) if any(x<0 for x in trades) else 0,
    }
    curve = pd.DataFrame({"step": range(len(equity)), "equity": equity})
    return stats, curve


st.set_page_config(page_title="OMEN Platform", page_icon="🎯", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
[data-testid='stMetric'] {background:#151b27;padding:12px;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

init_db()
seed_data()

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now(timezone.utc)

with st.sidebar:
    st.title("🎯 OMEN Platform")
    st.caption("Read-only analytics + paper trading")
    page = st.radio("Navigation", ["Dashboard", "Control Center", "Paper Trading", "Backtesting", "Data & Settings"])
    event_id = st.selectbox("Market", list(EVENTS.keys()), format_func=lambda x: EVENTS[x])
    if st.button("🔄 Generate Live Observation", use_container_width=True):
        add_live_observation(event_id)
        st.rerun()
    if st.button("🤖 Generate Predictor Call", use_container_width=True):
        generate_prediction(event_id)
        st.rerun()
    st.divider()
    st.success("Read-only OMEN mode")
    st.caption("No live orders. No account access. Paper trading only.")

obs, preds, trades = latest_data(event_id)
latest = obs.iloc[-1] if not obs.empty else None
latest_pred = preds.iloc[0] if not preds.empty else None

if page == "Dashboard":
    st.title("📈 OMEN Dashboard")
    st.caption(EVENTS[event_id])
    c1,c2,c3,c4,c5 = st.columns(5)
    price = float(latest.twap_price) if latest is not None else 0
    yes = float(latest.yes_multiplier) if latest is not None else 0
    no = float(latest.no_multiplier) if latest is not None else 0
    direction = str(latest.direction) if latest is not None else "N/A"
    signal = str(latest_pred.signal) if latest_pred is not None else "NEUTRAL"
    confidence = float(latest_pred.confidence) if latest_pred is not None else 0
    c1.metric("TWAP / YES Price", f"{price:.4f}")
    c2.metric("YES Multiplier", f"{yes:.4f}")
    c3.metric("NO Multiplier", f"{no:.4f}")
    c4.metric("Directional Flow", direction)
    c5.metric("Predictor", signal, f"{confidence:.0%} confidence" if latest_pred is not None else "No call")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("TWAP and Outcome Multipliers", "Volume"), vertical_spacing=0.12)
    fig.add_trace(go.Scatter(x=obs.timestamp, y=obs.twap_price, name="TWAP", line=dict(color="#00d4ff", width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=obs.timestamp, y=obs.yes_multiplier, name="YES", line=dict(color="#00d27a")), row=1, col=1)
    fig.add_trace(go.Scatter(x=obs.timestamp, y=obs.no_multiplier, name="NO", line=dict(color="#ff4b4b")), row=1, col=1)
    fig.add_trace(go.Bar(x=obs.timestamp, y=obs.volume, name="Volume", marker_color="#8d7bff"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=650, hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Recent Observations")
    st.dataframe(obs.tail(30).sort_values("timestamp", ascending=False), use_container_width=True)

elif page == "Control Center":
    st.title("🎮 Control Center")
    st.caption("Forecasting and signal fusion — analytics only")
    if st.button("Generate Forecast", type="primary"):
        result = generate_prediction(event_id)
        if result:
            st.success(f"Created {result['signal']} forecast at {result['confidence']:.0%} confidence")
        st.rerun()
    if latest_pred is not None:
        a,b,c,d = st.columns(4)
        a.metric("Current Price", f"{float(latest_pred.current_price):.4f}")
        b.metric("Forecast Price", f"{float(latest_pred.predicted_price):.4f}")
        c.metric("Signal", str(latest_pred.signal))
        d.metric("Confidence", f"{float(latest_pred.confidence):.1%}")
    if not preds.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=preds.timestamp, y=preds.current_price, name="Current", line=dict(color="#00d4ff")))
        fig.add_trace(go.Scatter(x=preds.timestamp, y=preds.predicted_price, name="Forecast", line=dict(color="#ffb000", dash="dash")))
        fig.update_layout(template="plotly_dark", height=420, title="Prediction History")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Latest Predictor Calls")
    st.dataframe(preds, use_container_width=True)

elif page == "Paper Trading":
    st.title("📄 PPTrader — Paper Trading")
    st.warning("Simulation only. This application does not place live trades or connect to brokerage accounts.")
    c1,c2,c3 = st.columns(3)
    strategy = c1.selectbox("Strategy", ["Strategy A: First Call", "Strategy B: Swap", "Strategy C: Predictor Only", "Strategy D: Dashboard Only", "Strategy E: Hybrid", "Strategy F: Custom Rule Engine"])
    capital = c2.number_input("Paper Capital", min_value=100.0, value=10000.0, step=100.0)
    size_pct = c3.slider("Position Size", 0.01, 0.50, 0.10, 0.01)
    c4,c5,c6 = st.columns(3)
    if c4.button("Open LONG Paper Position", use_container_width=True):
        open_paper_trade(event_id, strategy, "LONG", capital, size_pct)
        st.rerun()
    if c5.button("Open SHORT Paper Position", use_container_width=True):
        open_paper_trade(event_id, strategy, "SHORT", capital, size_pct)
        st.rerun()
    if c6.button("Close All Positions", use_container_width=True):
        close_all_positions(event_id)
        st.rerun()
    open_trades = trades[trades.status=="OPEN"] if not trades.empty else pd.DataFrame()
    closed = trades[trades.status=="CLOSED"] if not trades.empty else pd.DataFrame()
    x1,x2,x3,x4 = st.columns(4)
    x1.metric("Open Positions", len(open_trades))
    x2.metric("Closed Positions", len(closed))
    x3.metric("Closed P&L", f"${closed.pnl.fillna(0).sum():,.2f}" if not closed.empty else "$0.00")
    x4.metric("Win Rate", f"{(closed.pnl>0).mean()*100:.1f}%" if not closed.empty else "0.0%")
    st.subheader("Open Positions")
    st.dataframe(open_trades, use_container_width=True)
    st.subheader("Closed Positions")
    st.dataframe(closed, use_container_width=True)

elif page == "Backtesting":
    st.title("🧪 PPTrader — Backtesting")
    st.caption("Historical replay using the locally stored OMEN observation dataset.")
    c1,c2,c3,c4 = st.columns(4)
    strategy = c1.selectbox("Strategy", ["First Call", "Swap", "Predictor Only", "Dashboard Only", "Hybrid", "Custom Rule Engine"])
    capital = c2.number_input("Initial Capital", min_value=100.0, value=10000.0, step=100.0)
    size_pct = c3.slider("Size %", 0.01, 0.50, 0.10, 0.01, key="bt_size")
    commission = c4.number_input("Commission %", min_value=0.0, value=0.001, step=0.0005, format="%.4f")
    if st.button("Run Backtest", type="primary"):
        stats, curve = run_backtest(event_id, strategy, capital, size_pct, commission)
        if stats is None:
            st.error("Not enough observations to run a backtest.")
        else:
            st.session_state.backtest_stats = stats
            st.session_state.backtest_curve = curve
    if "backtest_stats" in st.session_state:
        stats = st.session_state.backtest_stats
        curve = st.session_state.backtest_curve
        a,b,c,d,e = st.columns(5)
        a.metric("Total Return", f"{stats['total_return']:.2f}%")
        b.metric("Sharpe", f"{stats['sharpe']:.2f}")
        c.metric("Max Drawdown", f"{stats['max_drawdown']:.2f}%")
        d.metric("Win Rate", f"{stats['win_rate']:.1f}%")
        e.metric("Trades", stats['total_trades'])
        fig = go.Figure(go.Scatter(x=curve.step, y=curve.equity, fill="tozeroy", line=dict(color="#00d4ff", width=3), name="Equity"))
        fig.update_layout(template="plotly_dark", title="Backtest Equity Curve", height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.json(stats)

else:
    st.title("⚙️ Data & Settings")
    st.subheader("Application Data")
    files = []
    for p in sorted(OUTPUT_DIR.glob("*.csv")):
        files.append({"File": p.name, "Size KB": round(p.stat().st_size/1024, 2)})
    st.dataframe(pd.DataFrame(files), use_container_width=True)
    st.subheader("Downloads")
    for p in sorted(OUTPUT_DIR.glob("*.csv")):
        st.download_button(f"Download {p.name}", p.read_bytes(), file_name=p.name, mime="text/csv")
    if st.button("Generate More Demo Data"):
        for eid in EVENTS:
            add_live_observation(eid)
            generate_prediction(eid)
        st.rerun()
    st.info("This app is intentionally read-only with respect to OMEN and any real trading account. It creates simulated data and paper trades locally.")

st.caption("OMEN Platform + PPTrader | Read-only analytics | Paper trading and backtesting only")
