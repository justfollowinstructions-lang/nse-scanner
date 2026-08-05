#!/usr/bin/env python3
"""
Signal Tags — watchlist categorization / badge system
=======================================================
Ported concept from the cup_test scanner's multi-category watchlist
(database.py: confirmed / on-the-verge / near-breakout / today / active /
historical), generalized to work across nse-scanner's 13 pattern
detectors instead of being Cup&Handle-specific.

cup_test derived its categories from Cup&Handle geometry (handle depth,
rim symmetry, etc). nse-scanner already has a pattern-agnostic composite
ranking system for exactly this purpose — score10 / tier, computed once
in scanner.py's main() from CANSLIM, RS percentile, risk:reward, volume
surge, Stage-2, Piotroski, pledge quality, etc — so tags here are derived
from THAT (same signal, any of the 14 pattern labels) rather than
re-deriving pattern-specific geometry.

Tags are additive: a single signal can carry several (e.g. a stock can be
both BUY_STRONG and MULTI_PATTERN and HIGH_CONVICTION at once). The
dashboard and watchlist both key off these tags for their category tabs.
"""

from __future__ import annotations

TAG_LABELS = {
    "HIGH_CONVICTION": "🌟 Very High Quality",
    "BUY_STRONG":      "💪 Buy Strong",
    "BUY_MODERATE":    "📊 Buy Moderate",
    "WATCH":           "👀 Watch",
    "MULTI_PATTERN":   "🔗 Multi-Pattern Confluence",
    "NEAR_BREAKOUT":   "⚡ Near Breakout",
    "ACTIVE_SETUP":    "🔥 Active / At Zone",
}

# Order tabs should render in on the dashboard / watchlist views
TAG_ORDER = ["HIGH_CONVICTION", "BUY_STRONG", "BUY_MODERATE", "MULTI_PATTERN",
             "ACTIVE_SETUP", "NEAR_BREAKOUT", "WATCH"]


def _f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key)
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def is_high_conviction(row: dict) -> bool:
    """
    The strict "very high good pattern" gate — this is the NEW,
    separate list the person asked for, analogous to cup_test's
    Confirmed-Breakout tier but generalized: instead of Cup&Handle
    geometry (handle depth / rim symmetry) it gates on the fields
    every one of nse-scanner's 14 pattern labels already produces.

    Deliberately a hard AND of several independent checks (not just a
    single high score10) so one maxed-out component can't carry a
    mediocre setup into the list — mirrors cup_test's approach of
    combining a numeric score with explicit hard requirements.
    """
    tier = str(row.get("tier") or "")
    score10   = _f(row, "score10")
    rr        = _f(row, "risk_reward")
    vol_surge = _f(row, "vol_surge")
    rs_pct    = _f(row, "rs_percentile")
    # data_completeness is a COUNT of CANSLIM sub-checks available for this
    # stock (denominator in score10's canslim_norm = cs/dc), not a 0-1
    # fraction — defaults to 7. Gate on "most checks were even available"
    # so a high score isn't resting on 1-2 lucky sub-checks.
    data_checks = _f(row, "data_completeness", 7.0)

    return (
        score10 >= 8.0
        and "BUY STRONG" in tier
        and rr >= 2.0
        and vol_surge >= 1.5
        and rs_pct >= 80
        and data_checks >= 5
    )


def compute_tags(row: dict) -> list[str]:
    """Return the list of tag keys (see TAG_LABELS) that apply to one
    signal row (a dict / pandas Series-like of the `signals` table
    columns, or the equivalent watchlist.json item)."""
    tags: list[str] = []
    tier = str(row.get("tier") or "")
    status = str(row.get("status") or "")
    converging = row.get("converging")

    if is_high_conviction(row):
        tags.append("HIGH_CONVICTION")

    if "BUY STRONG" in tier:
        tags.append("BUY_STRONG")
    elif "BUY MODERATE" in tier:
        tags.append("BUY_MODERATE")
    elif "WATCH" in tier:
        tags.append("WATCH")

    if converging is not None and str(converging).strip() not in ("", "nan", "None"):
        tags.append("MULTI_PATTERN")

    su = status.upper()
    if "READY" in su or "NEAR" in su or "APPROACHING" in su or "VERGE" in su:
        tags.append("NEAR_BREAKOUT")
    if "BREAKOUT ZONE" in su or "ACTIVE" in su or "CONFIRMED" in su:
        tags.append("ACTIVE_SETUP")

    return tags


def tag_dataframe(df):
    """Add a 'tags' column (comma-joined tag keys) to a signals
    DataFrame. Safe to call even on an empty DataFrame."""
    import pandas as pd
    if df is None or len(df) == 0:
        if df is not None and "tags" not in df.columns:
            df["tags"] = []
        return df
    df = df.copy()
    df["tags"] = df.apply(lambda r: ",".join(compute_tags(r.to_dict())), axis=1)
    return df


def high_conviction_table(df):
    """Return the strict 'Very High Quality' subset, best score first."""
    if df is None or len(df) == 0:
        return df
    mask = df.apply(lambda r: is_high_conviction(r.to_dict()), axis=1)
    out = df[mask].copy()
    if "score10" in out.columns:
        out = out.sort_values("score10", ascending=False)
    return out.reset_index(drop=True)
