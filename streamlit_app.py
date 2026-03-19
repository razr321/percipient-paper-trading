#!/usr/bin/env python3
"""
===============================================================================
  Percipient Digital Assets — Streamlit Paper Trading Dashboard
  Live simulation of 4 strategy tiers with full PnL, fee, and risk tracking.

  Strategies:
    1. Kalman Pairs (BTC/ETH)       — mean-reversion on spread z-score
    2. Dual Momentum (top 20)       — cross-sectional long/short
    3. Residual Momentum (top 20)   — factor-neutral residual ranking
    4. On-Chain Momentum (top 20)   — taker buy ratio signal

  Run:  streamlit run streamlit_app.py
===============================================================================
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings("ignore")

log = logging.getLogger("paper_trader")
log.setLevel(logging.INFO)
if not log.handlers:
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_ch)

# =============================================================================
#  PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Percipient Paper Trading",
    layout="wide",
    page_icon="\U0001f4ca",
    initial_sidebar_state="collapsed",
)

# Auto-refresh every 5 minutes (300,000 ms)
st_autorefresh(interval=300_000, limit=None, key="auto_refresh")

# =============================================================================
#  CONFIGURATION
# =============================================================================
STARTING_CAPITAL = 10_000.0

TOKENS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "TRXUSDT", "LINKUSDT",
    "DOTUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT", "NEARUSDT",
    "BCHUSDT", "FILUSDT", "MKRUSDT", "INJUSDT", "ARBUSDT",
]

# Fee model
TAKER_FEE = 0.00075       # 0.075% per side
MAKER_FEE = 0.00025       # 0.025% per side
SLIPPAGE = 0.0002         # 0.02% per trade
DEFAULT_FEE = TAKER_FEE   # assume taker
FUNDING_RATE_8H = 0.0001  # 0.01% per 8h for longs

# Strategy allocation
MAX_STRATEGY_ALLOC = 0.30  # 30% cap
BASE_ALLOC = 0.25          # 25% each

# Kalman pairs thresholds
KALMAN_ENTRY_Z = 1.5
KALMAN_EXIT_Z = 0.5
KALMAN_LOOKBACK = 60

# Dual momentum
DUAL_MOM_LOOKBACK = 30
DUAL_MOM_LONG_N = 5
DUAL_MOM_SHORT_N = 5

# Residual momentum
RESID_MOM_LOOKBACK = 30
RESID_MOM_N = 5

# On-chain momentum (taker buy ratio)
ONCHAIN_LONG_THRESH = 0.55
ONCHAIN_SHORT_THRESH = 0.45

# Colors
CLR_ACCENT = "#00d4ff"
CLR_GREEN = "#69db7c"
CLR_RED = "#ff6b6b"
CLR_YELLOW = "#ffd43b"
CLR_PURPLE = "#cc5de8"
CLR_BG = "#0f1117"
CLR_CARD = "#1a1d27"
CLR_BORDER = "#3a3d4d"
CLR_TEXT = "#e0e0e0"
PALETTE = [CLR_ACCENT, CLR_RED, CLR_GREEN, CLR_YELLOW, CLR_PURPLE,
           "#ff922b", "#74c0fc", "#f783ac", "#20c997", "#845ef7"]

# =============================================================================
#  STATE FILE PATH (works on local + Streamlit Cloud)
# =============================================================================
def _get_state_file_path() -> Path:
    """Return writable path for state file."""
    # Try local directory first
    local = Path(__file__).parent / "paper_trade_state.json"
    try:
        local.parent.mkdir(parents=True, exist_ok=True)
        # Test write access
        test = local.parent / ".write_test"
        test.write_text("ok")
        test.unlink()
        return local
    except (OSError, PermissionError):
        pass
    # Fall back to temp directory (Streamlit Cloud)
    return Path(tempfile.gettempdir()) / "paper_trade_state.json"


STATE_FILE = _get_state_file_path()

# =============================================================================
#  DATA FETCHING (Binance Vision API)
# =============================================================================
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_klines(symbol: str, interval: str = "1d", days: int = 90) -> pd.DataFrame:
    """Fetch OHLCV klines from Binance Vision API."""
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000

    all_klines: list = []
    cur = start_ms
    for _ in range(100):
        params = {"symbol": symbol, "interval": interval,
                  "limit": 1000, "startTime": cur}
        try:
            resp = httpx.get(BINANCE_URL, params=params, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            log.warning(f"Fetch error {symbol}: {e}")
            break

        klines = resp.json()
        if not klines:
            break
        all_klines.extend(klines)
        cur = klines[-1][0] + 1
        if len(klines) < 1000 or cur >= now_ms:
            break
        time.sleep(0.05)

    if not all_klines:
        return pd.DataFrame()

    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "n_trades",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    df = pd.DataFrame(all_klines, columns=cols)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_volume", "taker_buy_quote_volume"]:
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_current_prices(symbols: tuple) -> dict:
    """Fetch current prices for a list of symbols via ticker endpoint."""
    prices = {}
    try:
        url = "https://data-api.binance.vision/api/v3/ticker/price"
        resp = httpx.get(url, timeout=15)
        resp.raise_for_status()
        all_tickers = {t["symbol"]: float(t["price"]) for t in resp.json()}
        for s in symbols:
            if s in all_tickers:
                prices[s] = all_tickers[s]
    except Exception as e:
        log.warning(f"Price fetch error: {e}")
    return prices


def fetch_daily_closes(symbols: list, days: int = 60) -> pd.DataFrame:
    """Build a close price matrix for multiple tokens."""
    closes = {}
    for sym in symbols:
        df = fetch_klines(sym, "1d", days=days)
        if df.empty or len(df) < 10:
            continue
        ts = df.set_index("timestamp")["close"]
        ts.index = ts.index.normalize()
        closes[sym] = ts
    if not closes:
        return pd.DataFrame()
    return pd.DataFrame(closes).sort_index().ffill()


def fetch_taker_ratios(symbols: list, days: int = 7) -> dict:
    """Compute recent taker buy ratio for each symbol."""
    ratios = {}
    for sym in symbols:
        df = fetch_klines(sym, "1d", days=days)
        if df.empty or len(df) < 2:
            continue
        recent = df.tail(min(7, len(df)))
        total_vol = recent["volume"].sum()
        if total_vol > 0:
            ratios[sym] = recent["taker_buy_volume"].sum() / total_vol
    return ratios


# =============================================================================
#  TRADE & POSITION TYPES
# =============================================================================
@dataclass
class Trade:
    timestamp: str
    strategy: str
    symbol: str
    side: str
    size_usd: float
    entry_price: float
    exit_timestamp: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    fees: float = 0.0
    funding_paid: float = 0.0
    slippage_cost: float = 0.0
    holding_hours: float = 0.0
    closed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trade":
        return cls(**d)


@dataclass
class Position:
    strategy: str
    symbol: str
    side: str
    size_usd: float
    entry_price: float
    entry_time: str
    unrealized_pnl: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(**d)


# =============================================================================
#  PORTFOLIO STATE
# =============================================================================
class PortfolioState:
    """Tracks full paper trading state with persistence."""

    def __init__(self):
        self.starting_capital = STARTING_CAPITAL
        self.cash = STARTING_CAPITAL
        self.positions: list[Position] = []
        self.trade_history: list[Trade] = []
        self.equity_series: list[dict] = []
        self.daily_pnl: list[dict] = []
        self.strategy_allocs = {
            "kalman_pairs": BASE_ALLOC,
            "dual_momentum": BASE_ALLOC,
            "residual_momentum": BASE_ALLOC,
            "onchain_momentum": BASE_ALLOC,
        }
        self.total_trading_fees = 0.0
        self.total_funding_fees = 0.0
        self.total_slippage_costs = 0.0
        self.last_rebalance_date: Optional[str] = None
        self.last_update: Optional[str] = None
        self.strategy_wins: dict = {k: 0 for k in self.strategy_allocs}
        self.strategy_losses: dict = {k: 0 for k in self.strategy_allocs}
        self.strategy_gross_win: dict = {k: 0.0 for k in self.strategy_allocs}
        self.strategy_gross_loss: dict = {k: 0.0 for k in self.strategy_allocs}

    def save(self):
        """Persist state to JSON."""
        data = {
            "starting_capital": self.starting_capital,
            "cash": self.cash,
            "positions": [p.to_dict() for p in self.positions],
            "trade_history": [t.to_dict() for t in self.trade_history],
            "equity_series": self.equity_series,
            "daily_pnl": self.daily_pnl,
            "strategy_allocs": self.strategy_allocs,
            "total_trading_fees": self.total_trading_fees,
            "total_funding_fees": self.total_funding_fees,
            "total_slippage_costs": self.total_slippage_costs,
            "last_rebalance_date": self.last_rebalance_date,
            "last_update": self.last_update,
            "strategy_wins": self.strategy_wins,
            "strategy_losses": self.strategy_losses,
            "strategy_gross_win": self.strategy_gross_win,
            "strategy_gross_loss": self.strategy_gross_loss,
        }
        tmp = str(STATE_FILE) + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, str(STATE_FILE))
        except Exception as e:
            log.warning(f"Failed to save state: {e}")

    def load(self) -> bool:
        """Load state from JSON. Returns True if loaded successfully."""
        if not STATE_FILE.exists():
            return False
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            self.starting_capital = data.get("starting_capital", STARTING_CAPITAL)
            self.cash = data.get("cash", STARTING_CAPITAL)
            self.positions = [Position.from_dict(p) for p in data.get("positions", [])]
            self.trade_history = [Trade.from_dict(t) for t in data.get("trade_history", [])]
            self.equity_series = data.get("equity_series", [])
            self.daily_pnl = data.get("daily_pnl", [])
            self.strategy_allocs = data.get("strategy_allocs", self.strategy_allocs)
            self.total_trading_fees = data.get("total_trading_fees", 0.0)
            self.total_funding_fees = data.get("total_funding_fees", 0.0)
            self.total_slippage_costs = data.get("total_slippage_costs", 0.0)
            self.last_rebalance_date = data.get("last_rebalance_date")
            self.last_update = data.get("last_update")
            self.strategy_wins = data.get("strategy_wins", self.strategy_wins)
            self.strategy_losses = data.get("strategy_losses", self.strategy_losses)
            self.strategy_gross_win = data.get("strategy_gross_win", self.strategy_gross_win)
            self.strategy_gross_loss = data.get("strategy_gross_loss", self.strategy_gross_loss)
            return True
        except Exception as e:
            log.error(f"Failed to load state: {e}")
            return False

    def equity(self, prices: dict) -> float:
        """Current total equity = cash + unrealized PnL."""
        total = self.cash
        for pos in self.positions:
            if pos.symbol in prices:
                total += self._calc_upnl(pos, prices[pos.symbol])
        return total

    def _calc_upnl(self, pos: Position, current_price: float) -> float:
        """Calculate unrealized PnL for a position."""
        if pos.side == "long":
            ret = (current_price - pos.entry_price) / pos.entry_price
        else:
            ret = (pos.entry_price - current_price) / pos.entry_price
        return pos.size_usd * ret

    def open_position(self, strategy: str, symbol: str, side: str,
                      size_usd: float, price: float):
        """Open a new position. Deducts position size + fees from cash."""
        entry_fee = size_usd * DEFAULT_FEE
        slip_cost = size_usd * SLIPPAGE
        total_cost = entry_fee + slip_cost

        # Check sufficient cash
        required = size_usd + total_cost
        if self.cash < required:
            return  # skip if not enough cash

        # Deduct both position notional AND fees from cash
        self.cash -= required
        self.total_trading_fees += entry_fee
        self.total_slippage_costs += slip_cost

        now_str = datetime.now(timezone.utc).isoformat()
        pos = Position(
            strategy=strategy, symbol=symbol, side=side,
            size_usd=size_usd, entry_price=price, entry_time=now_str
        )
        self.positions.append(pos)

        trade = Trade(
            timestamp=now_str, strategy=strategy, symbol=symbol,
            side=side, size_usd=size_usd, entry_price=price,
            fees=entry_fee, slippage_cost=slip_cost
        )
        self.trade_history.append(trade)

    def close_position(self, pos: Position, price: float):
        """Close an existing position."""
        upnl = self._calc_upnl(pos, price)

        exit_fee = pos.size_usd * DEFAULT_FEE
        slip_cost = pos.size_usd * SLIPPAGE
        total_cost = exit_fee + slip_cost

        net_pnl = upnl - total_cost
        self.cash += (pos.size_usd + net_pnl)
        self.total_trading_fees += exit_fee
        self.total_slippage_costs += slip_cost

        strat = pos.strategy
        if net_pnl > 0:
            self.strategy_wins[strat] = self.strategy_wins.get(strat, 0) + 1
            self.strategy_gross_win[strat] = self.strategy_gross_win.get(strat, 0.0) + net_pnl
        else:
            self.strategy_losses[strat] = self.strategy_losses.get(strat, 0) + 1
            self.strategy_gross_loss[strat] = self.strategy_gross_loss.get(strat, 0.0) + abs(net_pnl)

        now_str = datetime.now(timezone.utc).isoformat()
        entry_dt = datetime.fromisoformat(pos.entry_time)
        hold_hours = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600

        for t in reversed(self.trade_history):
            if (t.strategy == pos.strategy and t.symbol == pos.symbol
                    and t.side == pos.side and not t.closed
                    and abs(t.entry_price - pos.entry_price) < 1e-8):
                t.exit_timestamp = now_str
                t.exit_price = price
                t.pnl = net_pnl
                t.fees += exit_fee
                t.slippage_cost += slip_cost
                t.holding_hours = hold_hours
                t.closed = True
                break

        self.positions.remove(pos)

    def apply_funding(self, prices: dict):
        """Apply funding rate to open positions (called every 8h)."""
        for pos in self.positions:
            funding = pos.size_usd * FUNDING_RATE_8H
            if pos.side == "long":
                self.cash -= funding
                self.total_funding_fees += funding
            else:
                self.cash += funding
                self.total_funding_fees -= funding

    def record_equity_snapshot(self, prices: dict):
        """Record current equity for time series."""
        eq = self.equity(prices)
        now_str = datetime.now(timezone.utc).isoformat()
        self.equity_series.append({
            "timestamp": now_str,
            "equity": round(eq, 2),
            "cash": round(self.cash, 2),
            "n_positions": len(self.positions),
        })
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if not self.daily_pnl or self.daily_pnl[-1]["date"] != today:
            prev_eq = self.daily_pnl[-1]["equity"] if self.daily_pnl else self.starting_capital
            self.daily_pnl.append({
                "date": today,
                "pnl": round(eq - prev_eq, 2),
                "equity": round(eq, 2),
            })
        else:
            self.daily_pnl[-1]["equity"] = round(eq, 2)
            if len(self.daily_pnl) > 1:
                self.daily_pnl[-1]["pnl"] = round(eq - self.daily_pnl[-2]["equity"], 2)

    def get_strategy_positions(self, strategy: str) -> list:
        return [p for p in self.positions if p.strategy == strategy]

    def get_strategy_pnl(self, strategy: str) -> float:
        return sum(t.pnl for t in self.trade_history if t.strategy == strategy and t.closed)

    def get_strategy_allocated_capital(self, strategy: str) -> float:
        return sum(p.size_usd for p in self.positions if p.strategy == strategy)


# =============================================================================
#  KELLY FRACTION CALCULATOR
# =============================================================================
def compute_kelly_fraction(wins: int, losses: int,
                           gross_win: float, gross_loss: float) -> float:
    """Compute quarter-Kelly fraction from historical trades."""
    total = wins + losses
    if total < 5:
        return BASE_ALLOC

    win_rate = wins / total
    if losses == 0 or gross_loss == 0:
        avg_ratio = 2.0
    else:
        avg_win = gross_win / max(wins, 1)
        avg_loss = gross_loss / max(losses, 1)
        avg_ratio = avg_win / avg_loss if avg_loss > 0 else 2.0

    kelly = win_rate - (1 - win_rate) / max(avg_ratio, 0.01)
    kelly = max(0.0, min(kelly, 1.0))
    return min(kelly * 0.25, MAX_STRATEGY_ALLOC)


def rebalance_allocations(state: PortfolioState):
    """Rebalance strategy allocations based on rolling Kelly."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if state.last_rebalance_date:
        last = datetime.strptime(state.last_rebalance_date, "%Y-%m-%d")
        if (datetime.strptime(today, "%Y-%m-%d") - last).days < 30:
            return

    new_allocs = {}
    for strat in state.strategy_allocs:
        w = state.strategy_wins.get(strat, 0)
        l = state.strategy_losses.get(strat, 0)
        gw = state.strategy_gross_win.get(strat, 0.0)
        gl = state.strategy_gross_loss.get(strat, 0.0)
        new_allocs[strat] = compute_kelly_fraction(w, l, gw, gl)

    total = sum(new_allocs.values())
    if total > 0:
        for k in new_allocs:
            new_allocs[k] = new_allocs[k] / total
    else:
        new_allocs = {k: BASE_ALLOC for k in state.strategy_allocs}

    state.strategy_allocs = new_allocs
    state.last_rebalance_date = today


# =============================================================================
#  STRATEGY IMPLEMENTATIONS
# =============================================================================
def strategy_kalman_pairs(state: PortfolioState, prices: dict,
                          price_history: pd.DataFrame):
    """Kalman Pairs: BTC/ETH spread z-score."""
    strat = "kalman_pairs"
    alloc = state.strategy_allocs.get(strat, BASE_ALLOC)
    budget = state.equity(prices) * alloc

    if "BTCUSDT" not in price_history.columns or "ETHUSDT" not in price_history.columns:
        return
    if len(price_history) < KALMAN_LOOKBACK + 5:
        return

    btc = price_history["BTCUSDT"].dropna()
    eth = price_history["ETHUSDT"].dropna()
    common = btc.index.intersection(eth.index)
    if len(common) < KALMAN_LOOKBACK + 5:
        return
    btc = btc.loc[common]
    eth = eth.loc[common]

    ratio = np.log(btc / eth)
    rolling_mean = ratio.rolling(KALMAN_LOOKBACK).mean()
    rolling_std = ratio.rolling(KALMAN_LOOKBACK).std()

    if rolling_std.iloc[-1] == 0 or np.isnan(rolling_std.iloc[-1]):
        return

    z_score = (ratio.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]

    current_positions = state.get_strategy_positions(strat)

    for pos in current_positions[:]:
        if abs(z_score) < KALMAN_EXIT_Z:
            if pos.symbol in prices:
                state.close_position(pos, prices[pos.symbol])

    current_positions = state.get_strategy_positions(strat)
    deployed = sum(p.size_usd for p in current_positions)

    if not current_positions:
        half = min(budget * 0.5, (budget - deployed) * 0.5)
        if half < 50:
            return

        if z_score < -KALMAN_ENTRY_Z:
            if "BTCUSDT" in prices and "ETHUSDT" in prices:
                state.open_position(strat, "BTCUSDT", "long", half, prices["BTCUSDT"])
                state.open_position(strat, "ETHUSDT", "short", half, prices["ETHUSDT"])
        elif z_score > KALMAN_ENTRY_Z:
            if "BTCUSDT" in prices and "ETHUSDT" in prices:
                state.open_position(strat, "BTCUSDT", "short", half, prices["BTCUSDT"])
                state.open_position(strat, "ETHUSDT", "long", half, prices["ETHUSDT"])


def strategy_dual_momentum(state: PortfolioState, prices: dict,
                           price_history: pd.DataFrame):
    """Dual Momentum: Rank tokens by 30-day return, long top 5, short bottom 5."""
    strat = "dual_momentum"
    alloc = state.strategy_allocs.get(strat, BASE_ALLOC)
    budget = state.equity(prices) * alloc

    if len(price_history) < DUAL_MOM_LOOKBACK + 5:
        return

    returns = {}
    for sym in price_history.columns:
        series = price_history[sym].dropna()
        if len(series) < DUAL_MOM_LOOKBACK + 1:
            continue
        ret = (series.iloc[-1] / series.iloc[-DUAL_MOM_LOOKBACK]) - 1
        if np.isfinite(ret):
            returns[sym] = ret

    if len(returns) < DUAL_MOM_LONG_N + DUAL_MOM_SHORT_N:
        return

    sorted_tokens = sorted(returns.items(), key=lambda x: x[1], reverse=True)
    long_targets = {t[0] for t in sorted_tokens[:DUAL_MOM_LONG_N]}
    short_targets = {t[0] for t in sorted_tokens[-DUAL_MOM_SHORT_N:]}

    for pos in state.get_strategy_positions(strat)[:]:
        should_close = False
        if pos.side == "long" and pos.symbol not in long_targets:
            should_close = True
        elif pos.side == "short" and pos.symbol not in short_targets:
            should_close = True
        if should_close and pos.symbol in prices:
            state.close_position(pos, prices[pos.symbol])

    current_syms = {(p.symbol, p.side) for p in state.get_strategy_positions(strat)}
    per_position = budget / (DUAL_MOM_LONG_N + DUAL_MOM_SHORT_N)
    per_position = max(per_position, 0)

    if per_position < 20:
        return

    for sym in long_targets:
        if (sym, "long") not in current_syms and sym in prices:
            deployed = state.get_strategy_allocated_capital(strat)
            if deployed + per_position <= budget * 1.1:
                state.open_position(strat, sym, "long", per_position, prices[sym])

    for sym in short_targets:
        if (sym, "short") not in current_syms and sym in prices:
            deployed = state.get_strategy_allocated_capital(strat)
            if deployed + per_position <= budget * 1.1:
                state.open_position(strat, sym, "short", per_position, prices[sym])


def strategy_residual_momentum(state: PortfolioState, prices: dict,
                               price_history: pd.DataFrame):
    """Residual Momentum: Token return minus BTC return, long top 5."""
    strat = "residual_momentum"
    alloc = state.strategy_allocs.get(strat, BASE_ALLOC)
    budget = state.equity(prices) * alloc

    if "BTCUSDT" not in price_history.columns:
        return
    if len(price_history) < RESID_MOM_LOOKBACK + 5:
        return

    btc_series = price_history["BTCUSDT"].dropna()
    if len(btc_series) < RESID_MOM_LOOKBACK + 1:
        return
    btc_ret = (btc_series.iloc[-1] / btc_series.iloc[-RESID_MOM_LOOKBACK]) - 1

    residuals = {}
    for sym in price_history.columns:
        if sym == "BTCUSDT":
            continue
        series = price_history[sym].dropna()
        if len(series) < RESID_MOM_LOOKBACK + 1:
            continue
        token_ret = (series.iloc[-1] / series.iloc[-RESID_MOM_LOOKBACK]) - 1
        resid = token_ret - btc_ret
        if np.isfinite(resid):
            residuals[sym] = resid

    if len(residuals) < RESID_MOM_N:
        return

    sorted_resid = sorted(residuals.items(), key=lambda x: x[1], reverse=True)
    long_targets = {t[0] for t in sorted_resid[:RESID_MOM_N]}

    for pos in state.get_strategy_positions(strat)[:]:
        if pos.symbol not in long_targets and pos.symbol in prices:
            state.close_position(pos, prices[pos.symbol])

    current_syms = {p.symbol for p in state.get_strategy_positions(strat)}
    per_position = budget / RESID_MOM_N
    if per_position < 20:
        return

    for sym in long_targets:
        if sym not in current_syms and sym in prices:
            deployed = state.get_strategy_allocated_capital(strat)
            if deployed + per_position <= budget * 1.1:
                state.open_position(strat, sym, "long", per_position, prices[sym])


def strategy_onchain_momentum(state: PortfolioState, prices: dict,
                              taker_ratios: dict):
    """On-Chain Momentum: Taker buy ratio signals."""
    strat = "onchain_momentum"
    alloc = state.strategy_allocs.get(strat, BASE_ALLOC)
    budget = state.equity(prices) * alloc

    if not taker_ratios:
        return

    long_targets = {s for s, r in taker_ratios.items() if r > ONCHAIN_LONG_THRESH}
    short_targets = {s for s, r in taker_ratios.items() if r < ONCHAIN_SHORT_THRESH}

    for pos in state.get_strategy_positions(strat)[:]:
        should_close = False
        if pos.side == "long" and pos.symbol not in long_targets:
            should_close = True
        elif pos.side == "short" and pos.symbol not in short_targets:
            should_close = True
        if should_close and pos.symbol in prices:
            state.close_position(pos, prices[pos.symbol])

    total_targets = len(long_targets) + len(short_targets)
    if total_targets == 0:
        return
    per_position = min(budget / max(total_targets, 1), budget * 0.15)
    if per_position < 20:
        return

    current_syms = {(p.symbol, p.side) for p in state.get_strategy_positions(strat)}

    for sym in long_targets:
        if (sym, "long") not in current_syms and sym in prices:
            deployed = state.get_strategy_allocated_capital(strat)
            if deployed + per_position <= budget * 1.1:
                state.open_position(strat, sym, "long", per_position, prices[sym])

    for sym in short_targets:
        if (sym, "short") not in current_syms and sym in prices:
            deployed = state.get_strategy_allocated_capital(strat)
            if deployed + per_position <= budget * 1.1:
                state.open_position(strat, sym, "short", per_position, prices[sym])


# =============================================================================
#  METRICS CALCULATION
# =============================================================================
def compute_metrics(state: PortfolioState, prices: dict) -> dict:
    """Compute portfolio and per-strategy metrics."""
    eq = state.equity(prices)
    total_pnl = eq - state.starting_capital
    total_fees = state.total_trading_fees + state.total_funding_fees + state.total_slippage_costs

    eq_values = [e["equity"] for e in state.equity_series]
    if len(eq_values) < 2:
        eq_values = [state.starting_capital, eq]

    eq_arr = np.array(eq_values)
    peak = np.maximum.accumulate(eq_arr)
    drawdowns = (eq_arr - peak) / peak
    max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0

    daily_rets = []
    for i in range(1, len(eq_arr)):
        if eq_arr[i - 1] > 0:
            daily_rets.append((eq_arr[i] / eq_arr[i - 1]) - 1)
    daily_rets = np.array(daily_rets) if daily_rets else np.array([0.0])
    sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(365)) if np.std(daily_rets) > 0 else 0.0

    ann_ret = float(np.mean(daily_rets) * 365) if len(daily_rets) > 0 else 0.0
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0.001 else 0.0

    strat_metrics = {}
    for strat in state.strategy_allocs:
        s_pnl = state.get_strategy_pnl(strat)
        s_upnl = 0.0
        for pos in state.get_strategy_positions(strat):
            if pos.symbol in prices:
                s_upnl += state._calc_upnl(pos, prices[pos.symbol])

        s_total = s_pnl + s_upnl
        w = state.strategy_wins.get(strat, 0)
        l = state.strategy_losses.get(strat, 0)
        total_trades = w + l
        win_rate = w / total_trades if total_trades > 0 else 0.0
        gw = state.strategy_gross_win.get(strat, 0.0)
        gl = state.strategy_gross_loss.get(strat, 0.0)
        profit_factor = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)
        avg_trade = s_pnl / total_trades if total_trades > 0 else 0.0

        s_sharpe = 0.0
        if total_trades >= 5:
            strat_trades = [t for t in state.trade_history if t.strategy == strat and t.closed]
            if strat_trades:
                pnls = np.array([t.pnl for t in strat_trades])
                s_sharpe = float(
                    np.mean(pnls) / np.std(pnls) * np.sqrt(365 / max(1, len(pnls)))
                ) if np.std(pnls) > 0 else 0.0

        kelly = compute_kelly_fraction(w, l, gw, gl)

        strat_metrics[strat] = {
            "realized_pnl": round(s_pnl, 2),
            "unrealized_pnl": round(s_upnl, 2),
            "total_pnl": round(s_total, 2),
            "n_trades": total_trades,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(min(profit_factor, 99.9), 2),
            "avg_trade": round(avg_trade, 2),
            "sharpe": round(s_sharpe, 2),
            "alloc": round(state.strategy_allocs.get(strat, BASE_ALLOC), 4),
            "kelly": round(kelly, 4),
            "n_positions": len(state.get_strategy_positions(strat)),
        }

    open_positions = []
    for pos in state.positions:
        curr = prices.get(pos.symbol, pos.entry_price)
        upnl = state._calc_upnl(pos, curr)
        open_positions.append({
            "symbol": pos.symbol,
            "strategy": pos.strategy,
            "side": pos.side,
            "size_usd": round(pos.size_usd, 2),
            "entry_price": pos.entry_price,
            "current_price": curr,
            "unrealized_pnl": round(upnl, 2),
            "entry_time": pos.entry_time,
        })

    # Compute total realized vs unrealized across all strategies
    total_realized = sum(state.get_strategy_pnl(s) for s in state.strategy_allocs)
    total_unrealized = 0.0
    for pos in state.positions:
        if pos.symbol in prices:
            total_unrealized += state._calc_upnl(pos, prices[pos.symbol])

    return {
        "equity": round(eq, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl / state.starting_capital * 100, 2),
        "realized_pnl": round(total_realized, 2),
        "unrealized_pnl": round(total_unrealized, 2),
        "total_fees": round(total_fees, 2),
        "trading_fees": round(state.total_trading_fees, 2),
        "funding_fees": round(state.total_funding_fees, 2),
        "slippage_costs": round(state.total_slippage_costs, 2),
        "net_pnl": round(total_pnl, 2),
        "max_drawdown": round(max_dd, 4),
        "sharpe": round(sharpe, 2),
        "calmar": round(calmar, 2),
        "n_positions": len(state.positions),
        "n_trades": len([t for t in state.trade_history if t.closed]),
        "cash": round(state.cash, 2),
        "strategies": strat_metrics,
        "open_positions": open_positions,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
#  PLOTLY CHARTS
# =============================================================================
CHART_LAYOUT = dict(
    paper_bgcolor=CLR_BG,
    plot_bgcolor=CLR_CARD,
    font=dict(color=CLR_TEXT, family="monospace", size=12),
    margin=dict(l=50, r=30, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor=CLR_BORDER, zerolinecolor=CLR_BORDER),
    yaxis=dict(gridcolor=CLR_BORDER, zerolinecolor=CLR_BORDER),
)


def build_equity_chart(state: PortfolioState) -> go.Figure:
    """Build equity curve chart with Plotly."""
    fig = go.Figure()

    if len(state.equity_series) < 2:
        fig.add_annotation(text="Waiting for data...", showarrow=False,
                           font=dict(size=16, color=CLR_TEXT),
                           xref="paper", yref="paper", x=0.5, y=0.5)
        fig.update_layout(**CHART_LAYOUT, title="Portfolio Equity Curve", height=350)
        return fig

    timestamps = [pd.Timestamp(e["timestamp"]) for e in state.equity_series]
    equities = [e["equity"] for e in state.equity_series]

    fig.add_trace(go.Scatter(
        x=timestamps, y=equities, mode="lines",
        line=dict(color=CLR_ACCENT, width=2),
        name="Portfolio Equity",
        fill="tozeroy", fillcolor="rgba(0, 212, 255, 0.08)",
    ))

    fig.add_hline(
        y=STARTING_CAPITAL, line_dash="dash",
        line_color=CLR_BORDER, annotation_text=f"Start ${STARTING_CAPITAL:,.0f}",
        annotation_font_color=CLR_TEXT,
    )

    fig.update_layout(
        **CHART_LAYOUT,
        title="Portfolio Equity Curve",
        yaxis_title="Equity ($)",
        height=350,
        hovermode="x unified",
    )
    return fig


def build_drawdown_chart(state: PortfolioState) -> go.Figure:
    """Build drawdown chart."""
    fig = go.Figure()

    if len(state.equity_series) < 2:
        fig.update_layout(**CHART_LAYOUT, title="Drawdown", height=250)
        return fig

    timestamps = [pd.Timestamp(e["timestamp"]) for e in state.equity_series]
    eq_arr = np.array([e["equity"] for e in state.equity_series])
    peak = np.maximum.accumulate(eq_arr)
    dd = ((eq_arr - peak) / peak) * 100

    fig.add_trace(go.Scatter(
        x=timestamps, y=dd, mode="lines",
        line=dict(color=CLR_RED, width=1.5),
        fill="tozeroy", fillcolor="rgba(255, 107, 107, 0.15)",
        name="Drawdown %",
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title="Drawdown (%)",
        yaxis_title="Drawdown %",
        height=250,
        hovermode="x unified",
    )
    return fig


def build_rolling_sharpe_chart(state: PortfolioState, window: int = 30) -> go.Figure:
    """Build rolling Sharpe ratio chart."""
    fig = go.Figure()

    if len(state.equity_series) < window + 2:
        fig.add_annotation(text=f"Need {window}+ data points", showarrow=False,
                           font=dict(size=14, color=CLR_TEXT),
                           xref="paper", yref="paper", x=0.5, y=0.5)
        fig.update_layout(**CHART_LAYOUT, title=f"Rolling {window}-period Sharpe", height=250)
        return fig

    timestamps = [pd.Timestamp(e["timestamp"]) for e in state.equity_series]
    eq_arr = np.array([e["equity"] for e in state.equity_series])

    rets = np.diff(eq_arr) / eq_arr[:-1]
    rolling_sharpe = []
    rolling_ts = []
    for i in range(window, len(rets)):
        chunk = rets[i - window:i]
        if np.std(chunk) > 0:
            s = np.mean(chunk) / np.std(chunk) * np.sqrt(365)
        else:
            s = 0.0
        rolling_sharpe.append(s)
        rolling_ts.append(timestamps[i + 1])

    fig.add_trace(go.Scatter(
        x=rolling_ts, y=rolling_sharpe, mode="lines",
        line=dict(color=CLR_PURPLE, width=1.5),
        name=f"Sharpe ({window}p)",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color=CLR_BORDER)

    fig.update_layout(
        **CHART_LAYOUT,
        title=f"Rolling {window}-period Sharpe Ratio",
        yaxis_title="Sharpe Ratio",
        height=250,
        hovermode="x unified",
    )
    return fig


def build_fee_pie(metrics: dict) -> go.Figure:
    """Build fee breakdown pie chart."""
    labels = ["Trading Fees", "Funding Fees", "Slippage"]
    values = [
        max(metrics.get("trading_fees", 0), 0),
        max(metrics.get("funding_fees", 0), 0),
        max(metrics.get("slippage_costs", 0), 0),
    ]

    if sum(values) == 0:
        values = [1, 1, 1]  # placeholder

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=[CLR_ACCENT, CLR_YELLOW, CLR_RED]),
        textfont=dict(color=CLR_TEXT),
        hole=0.4,
    ))
    fig.update_layout(
        paper_bgcolor=CLR_BG,
        plot_bgcolor=CLR_CARD,
        font=dict(color=CLR_TEXT, family="monospace", size=12),
        margin=dict(l=50, r=30, t=40, b=40),
        title="Fee Breakdown",
        height=300,
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    return fig


def build_allocation_pie(metrics: dict) -> go.Figure:
    """Build strategy allocation pie chart."""
    strat_names = {
        "kalman_pairs": "Kalman Pairs",
        "dual_momentum": "Dual Momentum",
        "residual_momentum": "Residual Mom.",
        "onchain_momentum": "On-Chain Mom.",
    }
    labels = []
    values = []
    for key, name in strat_names.items():
        s = metrics.get("strategies", {}).get(key, {})
        labels.append(name)
        values.append(s.get("alloc", 0.25))

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=PALETTE[:4]),
        textfont=dict(color=CLR_TEXT),
        hole=0.4,
    ))
    fig.update_layout(
        paper_bgcolor=CLR_BG,
        plot_bgcolor=CLR_CARD,
        font=dict(color=CLR_TEXT, family="monospace", size=12),
        margin=dict(l=50, r=30, t=40, b=40),
        title="Strategy Allocation",
        height=300,
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    return fig


# =============================================================================
#  MAIN TRADING ENGINE TICK
# =============================================================================
def run_trading_tick(state: PortfolioState) -> dict:
    """Run one tick of the trading engine: fetch data, run strategies, compute metrics."""
    # Fetch current prices
    prices = fetch_current_prices(tuple(TOKENS))
    if not prices:
        st.warning("Could not fetch prices from Binance. Retrying next cycle.")
        return compute_metrics(state, {})

    # Fetch daily close history
    price_history = fetch_daily_closes(TOKENS, days=90)

    # Fetch taker buy ratios
    taker_ratios = fetch_taker_ratios(TOKENS, days=7)

    # Rebalance allocations (monthly)
    rebalance_allocations(state)

    # Run strategies
    if not price_history.empty:
        strategy_kalman_pairs(state, prices, price_history)
        strategy_dual_momentum(state, prices, price_history)
        strategy_residual_momentum(state, prices, price_history)
    strategy_onchain_momentum(state, prices, taker_ratios)

    # Apply funding (simplified: apply proportionally each tick)
    # In reality this would be every 8h; we approximate
    if state.last_update:
        try:
            last_dt = datetime.fromisoformat(state.last_update)
            hours_elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
            if hours_elapsed >= 8:
                state.apply_funding(prices)
        except Exception:
            pass

    # Record equity snapshot
    state.record_equity_snapshot(prices)

    # Update state
    state.last_update = datetime.now(timezone.utc).isoformat()

    # Persist
    state.save()

    return compute_metrics(state, prices)


# =============================================================================
#  STREAMLIT DASHBOARD
# =============================================================================
def main():
    # Initialize state
    if "portfolio_state" not in st.session_state:
        state = PortfolioState()
        loaded = state.load()
        if not loaded:
            log.info("No saved state found; starting fresh with $10,000.")
        st.session_state["portfolio_state"] = state
        st.session_state["initialized"] = True

    state: PortfolioState = st.session_state["portfolio_state"]

    # Run trading tick (fetches data, runs strategies)
    with st.spinner("Fetching live prices and running strategies..."):
        metrics = run_trading_tick(state)

    # Sync back to session state
    st.session_state["portfolio_state"] = state

    # =========================================================================
    #  HEADER
    # =========================================================================
    st.title("\U0001f4ca Percipient Digital Assets \u2014 Paper Trading Dashboard")
    last_update = state.last_update or "N/A"
    if last_update != "N/A":
        try:
            dt = datetime.fromisoformat(last_update)
            last_update = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            pass
    st.caption(
        f"Last update: {last_update}  |  Starting Capital: $10,000  |  "
        f"Auto-refresh: 5 min  |  Positions: {metrics['n_positions']}  |  "
        f"Closed trades: {metrics['n_trades']}"
    )

    # =========================================================================
    #  TOP ROW — Key Metrics
    # =========================================================================
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    equity = metrics["equity"]
    total_pnl = metrics["total_pnl"]
    pnl_pct = metrics["total_pnl_pct"]
    realized_pnl = metrics.get("realized_pnl", 0)
    unrealized_pnl = metrics.get("unrealized_pnl", 0)
    total_fees = metrics["total_fees"]
    max_dd = metrics["max_drawdown"]
    sharpe = metrics["sharpe"]

    col1.metric("Portfolio Equity", f"${equity:,.2f}", f"{pnl_pct:+.1f}%")
    col2.metric("Realized PnL", f"${realized_pnl:+,.2f}", "Closed trades only")
    col3.metric("Unrealized PnL", f"${unrealized_pnl:+,.2f}", f"{len(metrics.get('open_positions', []))} open positions")
    col4.metric("Total Fees", f"${total_fees:,.2f}", f"Trading + Slip")
    col5.metric("Max Drawdown", f"{max_dd:.2%}", "From peak")
    col6.metric("Sharpe (ann.)", f"{sharpe:.2f}", f"Calmar: {metrics['calmar']:.2f}")

    # Cash available
    st.caption(f"Cash: ${metrics.get('cash', 0):,.2f} | Total PnL (Realized + Unrealized): ${total_pnl:+,.2f}")

    st.divider()

    # =========================================================================
    #  EQUITY CURVE
    # =========================================================================
    st.plotly_chart(build_equity_chart(state), use_container_width=True, key="equity_chart")

    # =========================================================================
    #  STRATEGY PERFORMANCE TABLE
    # =========================================================================
    st.subheader("Strategy Performance")

    strat_display = {
        "kalman_pairs": "Kalman Pairs (BTC/ETH)",
        "dual_momentum": "Dual Momentum",
        "residual_momentum": "Residual Momentum",
        "onchain_momentum": "On-Chain Momentum",
    }

    strat_rows = []
    for key, display_name in strat_display.items():
        s = metrics["strategies"].get(key, {})
        strat_rows.append({
            "Strategy": display_name,
            "Total PnL": f"${s.get('total_pnl', 0):+,.2f}",
            "Realized": f"${s.get('realized_pnl', 0):+,.2f}",
            "Unrealized": f"${s.get('unrealized_pnl', 0):+,.2f}",
            "Sharpe": f"{s.get('sharpe', 0):.2f}",
            "Win Rate": f"{s.get('win_rate', 0):.0%}",
            "Profit Factor": f"{s.get('profit_factor', 0):.2f}",
            "Allocation": f"{s.get('alloc', 0.25):.0%}",
            "Kelly": f"{s.get('kelly', 0.25):.1%}",
            "Trades": s.get("n_trades", 0),
            "Positions": s.get("n_positions", 0),
        })

    strat_df = pd.DataFrame(strat_rows)
    st.dataframe(strat_df, use_container_width=True, hide_index=True)

    # =========================================================================
    #  TWO COLUMNS: Open Positions | Fee & Allocation Charts
    # =========================================================================
    st.divider()
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.subheader("Open Positions")
        if metrics["open_positions"]:
            pos_data = []
            for p in metrics["open_positions"]:
                pos_data.append({
                    "Symbol": p["symbol"],
                    "Strategy": strat_display.get(p["strategy"], p["strategy"]),
                    "Side": p["side"].upper(),
                    "Size ($)": f"${p['size_usd']:,.2f}",
                    "Entry Price": f"{p['entry_price']:.4f}",
                    "Current Price": f"{p['current_price']:.4f}",
                    "Unrealized PnL": f"${p['unrealized_pnl']:+,.2f}",
                })
            pos_df = pd.DataFrame(pos_data)
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions.")

    with right_col:
        tab1, tab2 = st.tabs(["Fee Breakdown", "Allocation"])
        with tab1:
            st.plotly_chart(build_fee_pie(metrics), use_container_width=True, key="fee_pie")
        with tab2:
            st.plotly_chart(build_allocation_pie(metrics), use_container_width=True, key="alloc_pie")

    # =========================================================================
    #  RECENT TRADES TABLE
    # =========================================================================
    st.divider()
    st.subheader("Recent Trades (Last 30)")

    closed_trades = [t for t in state.trade_history if t.closed]
    recent = closed_trades[-30:] if len(closed_trades) > 30 else closed_trades
    recent = list(reversed(recent))  # most recent first

    if recent:
        trade_rows = []
        for t in recent:
            ts = t.exit_timestamp or t.timestamp
            try:
                dt = datetime.fromisoformat(ts)
                ts_fmt = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                ts_fmt = ts[:16]

            trade_rows.append({
                "Time": ts_fmt,
                "Strategy": strat_display.get(t.strategy, t.strategy),
                "Symbol": t.symbol,
                "Side": t.side.upper(),
                "Size ($)": f"${t.size_usd:,.2f}",
                "Entry": f"{t.entry_price:.4f}",
                "Exit": f"{t.exit_price:.4f}" if t.exit_price else "-",
                "PnL": f"${t.pnl:+,.2f}",
                "Fees": f"${t.fees:,.2f}",
                "Hold (h)": f"{t.holding_hours:.1f}",
            })
        trade_df = pd.DataFrame(trade_rows)
        st.dataframe(trade_df, use_container_width=True, hide_index=True)
    else:
        st.info("No closed trades yet.")

    # =========================================================================
    #  RISK METRICS CHARTS
    # =========================================================================
    st.divider()
    st.subheader("Risk Metrics")

    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:
        st.plotly_chart(build_drawdown_chart(state), use_container_width=True, key="dd_chart")

    with risk_col2:
        st.plotly_chart(build_rolling_sharpe_chart(state, window=30),
                        use_container_width=True, key="sharpe_chart")

    # =========================================================================
    #  DAILY PNL BAR CHART
    # =========================================================================
    if state.daily_pnl and len(state.daily_pnl) > 1:
        st.divider()
        st.subheader("Daily PnL")

        dates = [d["date"] for d in state.daily_pnl]
        pnls = [d["pnl"] for d in state.daily_pnl]
        colors = [CLR_GREEN if p >= 0 else CLR_RED for p in pnls]

        fig_daily = go.Figure(go.Bar(
            x=dates, y=pnls,
            marker_color=colors,
            name="Daily PnL",
        ))
        fig_daily.update_layout(
            **CHART_LAYOUT,
            title="Daily PnL ($)",
            yaxis_title="PnL ($)",
            height=300,
            hovermode="x unified",
        )
        st.plotly_chart(fig_daily, use_container_width=True, key="daily_pnl_chart")

    # =========================================================================
    #  SIDEBAR — Controls
    # =========================================================================
    with st.sidebar:
        st.header("Controls")

        if st.button("Force Refresh Now", use_container_width=True):
            # Clear cached data to force fresh fetch
            fetch_klines.clear()
            fetch_current_prices.clear()
            st.rerun()

        st.divider()

        if st.button("Reset Portfolio", type="secondary", use_container_width=True):
            if STATE_FILE.exists():
                STATE_FILE.unlink()
            st.session_state.pop("portfolio_state", None)
            st.session_state.pop("initialized", None)
            fetch_klines.clear()
            fetch_current_prices.clear()
            st.rerun()

        st.divider()
        st.caption("State file location:")
        st.code(str(STATE_FILE), language=None)

        st.divider()
        st.caption(
            "Strategies: Kalman Pairs, Dual Momentum, "
            "Residual Momentum, On-Chain Momentum"
        )
        st.caption(
            f"Fee model: {TAKER_FEE*100:.3f}% taker + "
            f"{SLIPPAGE*100:.2f}% slippage + funding"
        )
        st.caption(f"Tokens tracked: {len(TOKENS)}")
        st.caption(f"Position sizing: Quarter-Kelly, max {MAX_STRATEGY_ALLOC:.0%}/strategy")


if __name__ == "__main__":
    main()
