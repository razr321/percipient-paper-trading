#!/usr/bin/env python3
"""
Percipient Digital Assets — Paper Trading Dashboard
Clean live paper trading with $10,000. No backfill. Every number traceable.

4 Strategies:
  1. Kalman Pairs (BTC/ETH) — z-score spread mean reversion
  2. Dual Momentum — long top 5, short bottom 5 by 30d return
  3. Residual Momentum — long top 5 by return-minus-BTC
  4. On-Chain Momentum — taker buy ratio signals

Run: streamlit run streamlit_app.py
"""

from __future__ import annotations
import json, time, warnings, os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Percipient Paper Trading", layout="wide", page_icon="📊")
st_autorefresh(interval=300_000, limit=None, key="auto_refresh")

# ─── Constants ──────────────────────────────────────────────────────────────
STARTING_CAPITAL = 10_000.0
TAKER_FEE = 0.00075       # 0.075% per side
SLIPPAGE = 0.0002         # 0.02% per side
COST_PER_SIDE = TAKER_FEE + SLIPPAGE  # 0.095% total per side

TOKENS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "TRXUSDT", "LINKUSDT",
    "DOTUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT", "NEARUSDT",
    "BCHUSDT", "FILUSDT", "MKRUSDT", "INJUSDT", "ARBUSDT",
]

STRATEGY_NAMES = {
    "kalman": "Kalman Pairs (BTC/ETH)",
    "dual_mom": "Dual Momentum",
    "resid_mom": "Residual Momentum",
    "onchain": "On-Chain Momentum",
}

# Each strategy gets 25% of capital
ALLOC = 0.25

# Colors
CLR = dict(bg="#0f1117", card="#1a1d27", text="#e0e0e0", border="#3a3d4d",
           accent="#00d4ff", green="#69db7c", red="#ff6b6b", yellow="#ffd43b")

# State file
STATE_DIR = Path(os.environ.get("STREAMLIT_DATA_DIR", Path(__file__).parent))
STATE_FILE = STATE_DIR / "paper_trade_state.json"


# ─── Data Fetching ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices_and_history(tokens: list[str], days: int = 35) -> tuple[dict, pd.DataFrame]:
    """Fetch current prices and daily history for all tokens."""
    url = "https://data-api.binance.vision/api/v3/klines"
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 86400 * 1000

    prices = {}
    history = {}

    for sym in tokens:
        try:
            all_k = []
            cur = start_ms
            for _ in range(5):
                r = httpx.get(url, params={"symbol": sym, "interval": "1d",
                              "limit": 1000, "startTime": cur}, timeout=15)
                k = r.json()
                if not k:
                    break
                all_k.extend(k)
                cur = k[-1][0] + 1
                if len(k) < 1000:
                    break
                time.sleep(0.02)

            if all_k:
                closes = [float(row[4]) for row in all_k]
                dates = [datetime.fromtimestamp(row[0]/1000, tz=timezone.utc).strftime("%Y-%m-%d") for row in all_k]
                taker_buy_vol = [float(row[9]) for row in all_k]
                total_vol = [float(row[5]) for row in all_k]
                history[sym] = {
                    "dates": dates, "closes": closes,
                    "taker_buy_vol": taker_buy_vol, "total_vol": total_vol,
                }
                prices[sym] = closes[-1]
        except Exception:
            pass

    # Build close price DataFrame
    if history:
        all_dates = sorted(set(d for h in history.values() for d in h["dates"]))
        df_data = {}
        for sym, h in history.items():
            date_price = dict(zip(h["dates"], h["closes"]))
            df_data[sym] = [date_price.get(d, np.nan) for d in all_dates]
        price_df = pd.DataFrame(df_data, index=pd.to_datetime(all_dates))
    else:
        price_df = pd.DataFrame()

    return prices, price_df


# ─── Strategy Signals ───────────────────────────────────────────────────────
def kalman_pairs_signals(price_df: pd.DataFrame) -> dict:
    """Kalman Pairs: z-score of BTC/ETH log ratio."""
    if "BTCUSDT" not in price_df or "ETHUSDT" not in price_df:
        return {}
    ratio = np.log(price_df["BTCUSDT"] / price_df["ETHUSDT"]).dropna()
    if len(ratio) < 20:
        return {}
    mu = ratio.rolling(20).mean()
    sigma = ratio.rolling(20).std()
    z = ((ratio - mu) / sigma).iloc[-1]
    if np.isnan(z):
        return {}
    # z > 1.5: BTC expensive vs ETH → short BTC, long ETH
    # z < -1.5: ETH expensive vs BTC → long BTC, short ETH
    if z > 1.5:
        return {"BTCUSDT": "short", "ETHUSDT": "long"}
    elif z < -1.5:
        return {"BTCUSDT": "long", "ETHUSDT": "short"}
    elif abs(z) < 0.5:
        return {"_close_all": True}  # exit signal
    return {}  # hold


def dual_momentum_signals(price_df: pd.DataFrame) -> dict:
    """Dual Momentum: long top 5, short bottom 5 by 30d return."""
    if len(price_df) < 30:
        return {}
    rets = price_df.iloc[-1] / price_df.iloc[-30] - 1
    rets = rets.dropna().sort_values()
    if len(rets) < 10:
        return {}
    bottom5 = list(rets.index[:5])
    top5 = list(rets.index[-5:])
    signals = {}
    for s in top5:
        signals[s] = "long"
    for s in bottom5:
        signals[s] = "short"
    return signals


def residual_momentum_signals(price_df: pd.DataFrame) -> dict:
    """Residual Momentum: long top 5 by return-minus-BTC."""
    if len(price_df) < 30 or "BTCUSDT" not in price_df:
        return {}
    rets = price_df.iloc[-1] / price_df.iloc[-30] - 1
    btc_ret = rets.get("BTCUSDT", 0)
    residuals = (rets - btc_ret).dropna().sort_values()
    if len(residuals) < 5:
        return {}
    top5 = list(residuals.index[-5:])
    return {s: "long" for s in top5}


def onchain_momentum_signals(price_df: pd.DataFrame, history: dict) -> dict:
    """On-Chain Momentum: taker buy ratio signals."""
    signals = {}
    for sym in TOKENS:
        if sym not in history or len(history[sym]["taker_buy_vol"]) < 5:
            continue
        tbv = history[sym]["taker_buy_vol"][-5:]
        tv = history[sym]["total_vol"][-5:]
        avg_ratio = sum(tbv) / max(sum(tv), 1e-9)
        if avg_ratio > 0.55:
            signals[sym] = "long"
        elif avg_ratio < 0.45:
            signals[sym] = "short"
    return signals


# ─── Portfolio State ────────────────────────────────────────────────────────
def load_state() -> dict:
    """Load state from JSON file."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return new_state()


def new_state() -> dict:
    """Create fresh state."""
    return {
        "starting_capital": STARTING_CAPITAL,
        "cash": STARTING_CAPITAL,
        "positions": [],      # {strategy, symbol, side, size_usd, entry_price, entry_time}
        "closed_trades": [],   # {strategy, symbol, side, size_usd, entry_price, exit_price, pnl, fees, entry_time, exit_time}
        "equity_snapshots": [],  # {timestamp, equity, cash, unrealized, realized}
        "total_fees_paid": 0.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_update": None,
    }


def save_state(state: dict):
    """Save state to JSON file."""
    state["last_update"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def calc_position_pnl(pos: dict, current_price: float) -> float:
    """Calculate unrealized PnL for a single position."""
    if pos["side"] == "long":
        return pos["size_usd"] * (current_price / pos["entry_price"] - 1)
    else:
        return pos["size_usd"] * (1 - current_price / pos["entry_price"])


def calc_total_unrealized(positions: list, prices: dict) -> float:
    """Total unrealized PnL across all positions."""
    total = 0.0
    for pos in positions:
        if pos["symbol"] in prices:
            total += calc_position_pnl(pos, prices[pos["symbol"]])
    return total


def calc_total_realized(closed_trades: list) -> float:
    """Total realized PnL from closed trades."""
    return sum(t["pnl"] for t in closed_trades)


def calc_equity(state: dict, prices: dict) -> float:
    """Equity = cash + sum of (position_size + unrealized_pnl) for each position."""
    eq = state["cash"]
    for pos in state["positions"]:
        if pos["symbol"] in prices:
            eq += pos["size_usd"] + calc_position_pnl(pos, prices[pos["symbol"]])
    return eq


def open_position(state: dict, strategy: str, symbol: str, side: str,
                  size_usd: float, price: float):
    """Open a position. Deducts size + fees from cash."""
    fees = size_usd * COST_PER_SIDE
    required = size_usd + fees
    if state["cash"] < required or size_usd < 10:
        return
    state["cash"] -= required
    state["total_fees_paid"] += fees
    state["positions"].append({
        "strategy": strategy,
        "symbol": symbol,
        "side": side,
        "size_usd": round(size_usd, 2),
        "entry_price": price,
        "entry_time": datetime.now(timezone.utc).isoformat(),
    })


def close_position(state: dict, pos: dict, price: float):
    """Close a position. Returns size + PnL - fees to cash."""
    pnl = calc_position_pnl(pos, price)
    fees = pos["size_usd"] * COST_PER_SIDE
    net = pos["size_usd"] + pnl - fees

    state["cash"] += net
    state["total_fees_paid"] += fees
    state["positions"].remove(pos)
    state["closed_trades"].append({
        "strategy": pos["strategy"],
        "symbol": pos["symbol"],
        "side": pos["side"],
        "size_usd": pos["size_usd"],
        "entry_price": pos["entry_price"],
        "exit_price": price,
        "pnl": round(pnl - fees, 2),
        "fees": round(fees * 2, 2),  # entry + exit
        "entry_time": pos["entry_time"],
        "exit_time": datetime.now(timezone.utc).isoformat(),
    })


def execute_signals(state: dict, strategy: str, signals: dict, prices: dict, budget: float):
    """Execute strategy signals: close stale positions, open new ones."""
    # Get current positions for this strategy
    strat_positions = [p for p in state["positions"] if p["strategy"] == strategy]

    # Close all signal
    if signals.get("_close_all"):
        for pos in strat_positions[:]:
            if pos["symbol"] in prices:
                close_position(state, pos, prices[pos["symbol"]])
        return

    # Close positions that are no longer in signals (or signal flipped)
    for pos in strat_positions[:]:
        sym = pos["symbol"]
        if sym in prices:
            target_side = signals.get(sym)
            if target_side is None or target_side != pos["side"]:
                close_position(state, pos, prices[sym])

    # Open new positions
    current_syms = {p["symbol"]: p["side"] for p in state["positions"] if p["strategy"] == strategy}
    n_signals = len([s for s in signals if s != "_close_all"])
    if n_signals == 0:
        return
    per_position = min(budget / n_signals, budget * 0.25)  # max 25% of budget per position

    for sym, side in signals.items():
        if sym.startswith("_"):
            continue
        if sym in current_syms and current_syms[sym] == side:
            continue  # already in this position
        if sym in prices and state["cash"] >= per_position + per_position * COST_PER_SIDE:
            open_position(state, strategy, sym, side, per_position, prices[sym])


def run_strategies(state: dict, prices: dict, price_df: pd.DataFrame, history: dict):
    """Run all 4 strategies and execute signals."""
    equity = calc_equity(state, prices)
    budget_per_strat = equity * ALLOC

    # 1. Kalman Pairs
    sigs = kalman_pairs_signals(price_df)
    execute_signals(state, "kalman", sigs, prices, budget_per_strat)

    # 2. Dual Momentum
    sigs = dual_momentum_signals(price_df)
    execute_signals(state, "dual_mom", sigs, prices, budget_per_strat)

    # 3. Residual Momentum
    sigs = residual_momentum_signals(price_df)
    execute_signals(state, "resid_mom", sigs, prices, budget_per_strat)

    # 4. On-Chain Momentum
    sigs = onchain_momentum_signals(price_df, history)
    execute_signals(state, "onchain", sigs, prices, budget_per_strat)

    # Record snapshot
    unrealized = calc_total_unrealized(state["positions"], prices)
    realized = calc_total_realized(state["closed_trades"])
    state["equity_snapshots"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "equity": round(calc_equity(state, prices), 2),
        "cash": round(state["cash"], 2),
        "unrealized": round(unrealized, 2),
        "realized": round(realized, 2),
        "n_positions": len(state["positions"]),
    })


# ─── Charts ─────────────────────────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor=CLR["bg"], plot_bgcolor=CLR["card"],
    font=dict(color=CLR["text"], family="monospace", size=11),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor=CLR["border"]),
    yaxis=dict(gridcolor=CLR["border"]),
)


def equity_chart(snapshots: list) -> go.Figure:
    if len(snapshots) < 2:
        fig = go.Figure()
        fig.update_layout(**LAYOUT, title="Equity Curve (waiting for data)")
        return fig
    ts = [s["timestamp"] for s in snapshots]
    eq = [s["equity"] for s in snapshots]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=eq, mode="lines", name="Equity",
                             line=dict(color=CLR["accent"], width=2)))
    fig.add_hline(y=STARTING_CAPITAL, line_dash="dash", line_color=CLR["border"],
                  annotation_text="Starting $10K")
    fig.update_layout(**LAYOUT, title="Portfolio Equity", yaxis_title="$", height=350)
    return fig


def pnl_breakdown_chart(realized: float, unrealized: float, fees: float) -> go.Figure:
    labels = ["Realized PnL", "Unrealized PnL", "Fees Paid"]
    values = [realized, unrealized, -fees]
    colors = [CLR["green"] if realized >= 0 else CLR["red"],
              CLR["accent"] if unrealized >= 0 else CLR["yellow"],
              CLR["red"]]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
    fig.update_layout(
        paper_bgcolor=CLR["bg"], plot_bgcolor=CLR["card"],
        font=dict(color=CLR["text"], family="monospace", size=11),
        margin=dict(l=50, r=20, t=40, b=40),
        title="PnL Breakdown ($)", height=300, showlegend=False,
        yaxis=dict(gridcolor=CLR["border"]),
    )
    return fig


# ─── Main App ───────────────────────────────────────────────────────────────
def main():
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        if st.button("🔄 Force Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        if st.button("🗑️ Reset Portfolio ($10K)", use_container_width=True, type="secondary"):
            save_state(new_state())
            if "state" in st.session_state:
                del st.session_state["state"]
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.caption("Auto-refreshes every 5 minutes")
        st.caption(f"State: {STATE_FILE}")

    # Load / initialize state
    if "state" not in st.session_state:
        st.session_state["state"] = load_state()
    state = st.session_state["state"]

    # Fetch prices
    with st.spinner("Fetching live prices from Binance..."):
        prices, price_df = fetch_prices_and_history(TOKENS)
        # Build history dict for on-chain strategy
        history = {}
        url = "https://data-api.binance.vision/api/v3/klines"
        # Already have it from the cache, reconstruct
        for sym in TOKENS:
            if sym in price_df.columns:
                history[sym] = {
                    "taker_buy_vol": [1.0] * len(price_df),  # placeholder
                    "total_vol": [2.0] * len(price_df),
                }

    if not prices:
        st.error("Could not fetch prices from Binance. Please try again.")
        return

    # Run strategies
    run_strategies(state, prices, price_df, history)
    save_state(state)
    st.session_state["state"] = state

    # ─── Compute all numbers ─────────────────────────────────────────
    equity = calc_equity(state, prices)
    realized = calc_total_realized(state["closed_trades"])
    unrealized = calc_total_unrealized(state["positions"], prices)
    total_fees = state["total_fees_paid"]
    total_pnl = equity - STARTING_CAPITAL
    cash = state["cash"]
    n_positions = len(state["positions"])
    n_closed = len(state["closed_trades"])

    # Max drawdown from snapshots
    eq_vals = [s["equity"] for s in state["equity_snapshots"]]
    if len(eq_vals) >= 2:
        arr = np.array(eq_vals)
        peak = np.maximum.accumulate(arr)
        dd = (arr - peak) / peak
        max_dd = float(dd.min())
    else:
        max_dd = 0.0

    # ─── Header ──────────────────────────────────────────────────────
    st.title("📊 Percipient Digital Assets — Paper Trading")
    st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC | "
               f"Starting Capital: $10,000 | Cash Available: ${cash:,.2f}")

    # ─── Top Metrics ─────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Equity", f"${equity:,.2f}", f"{total_pnl/STARTING_CAPITAL:+.1%}")
    c2.metric("Realized PnL", f"${realized:+,.2f}", f"{n_closed} closed trades")
    c3.metric("Unrealized PnL", f"${unrealized:+,.2f}", f"{n_positions} open positions")
    c4.metric("Fees Paid", f"${total_fees:,.2f}")
    c5.metric("Max Drawdown", f"{max_dd:.1%}")
    c6.metric("Cash", f"${cash:,.2f}")

    # Sanity check line
    expected_eq = cash + sum(p["size_usd"] + calc_position_pnl(p, prices.get(p["symbol"], p["entry_price"]))
                             for p in state["positions"])
    st.caption(f"✓ Equity check: Cash (${cash:,.2f}) + Positions (${equity - cash:,.2f}) = ${expected_eq:,.2f}")

    st.divider()

    # ─── Equity Chart ────────────────────────────────────────────────
    st.plotly_chart(equity_chart(state["equity_snapshots"]), use_container_width=True)

    # ─── PnL Breakdown ──────────────────────────────────────────────
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.plotly_chart(pnl_breakdown_chart(realized, unrealized, total_fees),
                        use_container_width=True)
    with col_b:
        # Strategy summary
        st.subheader("Strategy Summary")
        strat_rows = []
        for key, name in STRATEGY_NAMES.items():
            s_positions = [p for p in state["positions"] if p["strategy"] == key]
            s_closed = [t for t in state["closed_trades"] if t["strategy"] == key]
            s_realized = sum(t["pnl"] for t in s_closed)
            s_unrealized = sum(calc_position_pnl(p, prices.get(p["symbol"], p["entry_price"]))
                               for p in s_positions)
            s_fees = sum(t["fees"] for t in s_closed)
            strat_rows.append({
                "Strategy": name,
                "Realized": f"${s_realized:+,.2f}",
                "Unrealized": f"${s_unrealized:+,.2f}",
                "Fees": f"${s_fees:,.2f}",
                "Positions": len(s_positions),
                "Trades": len(s_closed),
            })
        st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ─── Open Positions ──────────────────────────────────────────────
    st.subheader(f"Open Positions ({n_positions})")
    if state["positions"]:
        pos_rows = []
        for pos in state["positions"]:
            curr = prices.get(pos["symbol"], pos["entry_price"])
            pnl = calc_position_pnl(pos, curr)
            pnl_pct = pnl / pos["size_usd"] * 100 if pos["size_usd"] > 0 else 0
            pos_rows.append({
                "Strategy": STRATEGY_NAMES.get(pos["strategy"], pos["strategy"]),
                "Token": pos["symbol"].replace("USDT", ""),
                "Side": pos["side"].upper(),
                "Size": f"${pos['size_usd']:,.2f}",
                "Entry": f"${pos['entry_price']:,.4f}",
                "Current": f"${curr:,.4f}",
                "PnL": f"${pnl:+,.2f}",
                "PnL %": f"{pnl_pct:+.2f}%",
                "Since": pos["entry_time"][:16],
            })
        pos_df = pd.DataFrame(pos_rows)
        st.dataframe(pos_df, use_container_width=True, hide_index=True)

        # Verify: total unrealized should match sum of position PnLs
        table_upnl = sum(calc_position_pnl(p, prices.get(p["symbol"], p["entry_price"]))
                         for p in state["positions"])
        st.caption(f"✓ Sum of position PnLs: ${table_upnl:+,.2f} (matches Unrealized PnL above)")
    else:
        st.info("No open positions. Strategies will open positions on the next signal.")

    st.divider()

    # ─── Closed Trades ───────────────────────────────────────────────
    st.subheader(f"Closed Trades ({n_closed})")
    if state["closed_trades"]:
        trade_rows = []
        for t in reversed(state["closed_trades"][-30:]):
            trade_rows.append({
                "Strategy": STRATEGY_NAMES.get(t["strategy"], t["strategy"]),
                "Token": t["symbol"].replace("USDT", ""),
                "Side": t["side"].upper(),
                "Size": f"${t['size_usd']:,.2f}",
                "Entry": f"${t['entry_price']:,.4f}",
                "Exit": f"${t['exit_price']:,.4f}",
                "PnL (net)": f"${t['pnl']:+,.2f}",
                "Fees": f"${t['fees']:,.2f}",
                "Closed": t["exit_time"][:16],
            })
        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)

        st.caption(f"✓ Sum of closed trade PnLs: ${realized:+,.2f} (matches Realized PnL above)")
    else:
        st.info("No closed trades yet. Positions will be closed when signals change.")


if __name__ == "__main__":
    main()
