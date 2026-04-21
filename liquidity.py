"""
Module 3: Liquidity & Market Impact Engine
============================================
Implements:
  - Amihud Illiquidity Ratio (Amihud, 2002)
  - Kyle's Lambda – price impact per unit volume
  - Adverse selection / Collin-Dufresne & Fos (2012) signature detection
  - Market Impact Curves (concave institutional footprint)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import linregress

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Amihud Illiquidity Ratio
# ─────────────────────────────────────────────────────────────────────────────

def compute_amihud(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Amihud (2002): ILLIQ_t = |R_t| / Volume_t
    Rolling average gives the per-symbol illiquidity time series.

    High ILLIQ → each unit of volume moves prices more → thin market.
    """
    df = df.sort_values("date").copy()
    df["return"]  = df.groupby("symbol")["price"].pct_change().abs()
    df["illiq"]   = df["return"] / df["quantity"].replace(0, np.nan)
    df["amihud"]  = (
        df.groupby("symbol")["illiq"]
          .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Kyle's Lambda (price impact coefficient)
# ─────────────────────────────────────────────────────────────────────────────

def compute_kyle_lambda(
    df: pd.DataFrame,
    symbol: str,
    window: int = 50,
) -> pd.DataFrame:
    """
    Kyle (1985): regress ΔP on signed volume to estimate lambda.
    ΔP_t = λ · Q_t + ε

    Rolling OLS over 'window' trades.

    High lambda → market is thick with informed traders.
    """
    sub = df[df["symbol"] == symbol].sort_values("date").copy()
    if "trade_sign" not in sub.columns:
        from modules.order_flow import classify_trade_sign
        sub = classify_trade_sign(sub)

    sub["delta_p"]     = sub["price"].diff()
    sub["signed_vol"]  = sub["quantity"] * sub["trade_sign"]

    lambdas = []
    for i in range(len(sub)):
        start = max(0, i - window + 1)
        chunk = sub.iloc[start : i + 1]
        x = chunk["signed_vol"].dropna().values
        y = chunk["delta_p"].dropna().values
        n = min(len(x), len(y))
        if n < 10:
            lambdas.append(np.nan)
            continue
        try:
            slope, _, _, _, _ = linregress(x[-n:], y[-n:])
            lambdas.append(slope)
        except Exception:
            lambdas.append(np.nan)

    sub["kyle_lambda"] = lambdas
    return sub


# ─────────────────────────────────────────────────────────────────────────────
#  Adverse Selection Risk (Collin-Dufresne & Fos, 2012)
# ─────────────────────────────────────────────────────────────────────────────

def compute_adverse_selection_risk(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    CDF (2012) logic: adverse selection is signalled when:
      - Trade size spikes
      - Bid-ask spread widens simultaneously
      - Price impact is above rolling average

    We proxy bid-ask spread using intra-window high-low / mid-price.
    Returns an 'adverse_selection_score' [0, 1].
    """
    df = df.sort_values("date").copy()

    # Proxy spread from intra-window price dispersion
    df["rolling_spread"] = (
        df.groupby("symbol")["price"]
          .transform(lambda s: (s.rolling(window).max() - s.rolling(window).min())
                               / s.rolling(window).mean())
    )

    # Volume spike: ratio to rolling mean
    df["vol_spike"] = (
        df.groupby("symbol")["quantity"]
          .transform(lambda s: s / s.rolling(window, min_periods=1).mean())
    )

    # Combine: adverse selection score normalised to [0,1]
    raw = (df["rolling_spread"].fillna(0) * 0.5 + (df["vol_spike"].fillna(1) - 1).clip(0) * 0.5)
    min_v, max_v = raw.min(), raw.max()
    if max_v > min_v:
        df["adverse_selection_score"] = (raw - min_v) / (max_v - min_v)
    else:
        df["adverse_selection_score"] = 0.0

    df["adverse_selection_flag"] = df["adverse_selection_score"] > 0.7
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Market Impact Curves
# ─────────────────────────────────────────────────────────────────────────────

def compute_market_impact_curve(
    df: pd.DataFrame,
    symbol: str,
    n_bins: int = 20,
) -> pd.DataFrame:
    """
    Institutional algo execution leaves a concave market impact signature:
    small trades have disproportionately high impact; large trades are
    well-hidden via TWAP/VWAP slicing.

    Bins trades by size and computes average absolute price impact per bin.
    Returns a DataFrame ready for plotting.
    """
    sub = df[df["symbol"] == symbol].copy()
    if len(sub) < 30:
        return pd.DataFrame()

    sub["abs_return"] = sub["price"].pct_change().abs()
    sub["size_bin"]   = pd.qcut(sub["quantity"], n_bins, duplicates="drop")

    curve = (
        sub.groupby("size_bin", observed=True)
           .agg(
               avg_qty=("quantity", "mean"),
               avg_impact=("abs_return", "mean"),
               count=("quantity", "count"),
           )
           .reset_index()
    )
    curve["avg_qty"] = curve["avg_qty"].astype(float)

    # Institutional concavity test: fit log model
    if len(curve) > 5:
        log_qty = np.log1p(curve["avg_qty"].values)
        impact  = curve["avg_impact"].values
        mask    = ~np.isnan(log_qty) & ~np.isnan(impact)
        if mask.sum() > 3:
            slope, intercept, r, _, _ = linregress(log_qty[mask], impact[mask])
            curve["fitted_impact"] = np.expm1(slope * log_qty + intercept)
            curve["concavity_r2"]  = round(r ** 2, 4)
        else:
            curve["fitted_impact"] = np.nan
            curve["concavity_r2"]  = np.nan

    return curve


# ─────────────────────────────────────────────────────────────────────────────
#  VWAP-based execution quality
# ─────────────────────────────────────────────────────────────────────────────

def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VWAP per symbol per day. Deviation from VWAP hints at
    informed trading (buying below / selling above VWAP consistently).
    """
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["date"]).dt.date
    df["pv"] = df["price"] * df["quantity"]

    vwap = (
        df.groupby(["symbol", "trade_date"])
          .apply(lambda g: g["pv"].sum() / g["quantity"].sum(), include_groups=False)
          .reset_index(name="vwap")
    )
    df = df.merge(vwap, on=["symbol", "trade_date"], how="left")
    df["vwap_deviation"] = (df["price"] - df["vwap"]) / df["vwap"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Per-symbol liquidity dashboard summary
# ─────────────────────────────────────────────────────────────────────────────

def liquidity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a per-symbol summary table combining Amihud, spread proxy, ADV.
    """
    df = compute_amihud(df)
    df = compute_vwap(df)

    summary = (
        df.groupby("symbol")
          .agg(
              avg_amihud    =("amihud", "mean"),
              avg_price     =("price", "mean"),
              total_volume  =("quantity", "sum"),
              avg_vwap_dev  =("vwap_deviation", lambda x: x.abs().mean()),
              n_trades      =("quantity", "count"),
          )
          .reset_index()
    )
    summary["liquidity_score"] = 1 / (1 + summary["avg_amihud"].fillna(0))
    return summary.sort_values("liquidity_score", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: run full Module 3 pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_liquidity_engine(df: pd.DataFrame, symbol: str = None) -> dict:
    """
    Run all Module 3 analytics.
    """
    df = compute_amihud(df)
    df = compute_adverse_selection_risk(df)
    df = compute_vwap(df)

    liq_summary = liquidity_summary(df)

    symbol = symbol or (df["symbol"].value_counts().idxmax() if not df.empty else "NABIL")
    kyle_df    = compute_kyle_lambda(df, symbol)
    impact_crv = compute_market_impact_curve(df, symbol)

    return {
        "enriched_df":    df,
        "liquidity_summary": liq_summary,
        "kyle_df":        kyle_df,
        "impact_curve":   impact_crv,
    }
