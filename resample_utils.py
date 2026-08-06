#!/usr/bin/env python3
"""
Resample Utils — derive weekly/monthly OHLCV from daily, locally
====================================================================
Ported directly from the cup_test scanner's downloader.py. This is the
fix for "why is it downloading weekly/monthly separately" — cup_test
never downloads weekly/monthly bars over the network at all. It
downloads daily bars ONCE, then derives weekly/monthly bars from the
already-cached daily data via pandas resampling, in-process, for free.

nse-scanner previously issued three full batched network downloads per
EOD update — one each for interval="1d", "1wk", "1mo" — which is 3x the
network time and 3x the chance of a rate-limit failure for no benefit,
since Yahoo's own weekly/monthly bars are themselves just aggregates of
the same daily bars. This module is used by data_updater.py (to
populate the multi-TF cache) and by dashboard_export.py (as the
fallback when a stock's weekly/monthly cache is empty for some reason)
so there is exactly one resampling implementation, not two slightly
different ones.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def resample_weekly(daily: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Weekly OHLCV, grouped by ISO calendar week (Monday-Sunday) rather
    than pandas' 'W-FRI' resample anchor.

    Why not plain `.resample("W-FRI", label="left", closed="left")`:
    that combination labels each week's bar with the *previous*
    Friday (a full week earlier than the week it actually
    summarizes) — e.g. the Mon Jan 1 - Fri Jan 5 week gets labeled
    "2023-12-29". This silently shifts every weekly candle back by
    about a week versus what any real charting platform (TradingView,
    NSE's own charts) shows for the same stock.

    This groups by (ISO year, ISO week) directly and labels each
    resulting bar with the first trading day actually present in that
    week — so a normal week is labeled Monday, and a week where Monday
    was an NSE holiday is correctly labeled with Tuesday (or whichever
    day trading actually resumed), rather than assuming Monday always
    exists.
    """
    if daily is None or daily.empty:
        return daily

    iso = daily.index.isocalendar()
    work = daily.copy()
    work["_iso_year"] = iso["year"].values
    work["_iso_week"] = iso["week"].values

    grouped = work.groupby(["_iso_year", "_iso_week"], sort=True)
    agg_kwargs = {"Open": ("Open", "first"), "High": ("High", "max"),
                  "Low": ("Low", "min"), "Close": ("Close", "last")}
    if "Volume" in work.columns:
        agg_kwargs["Volume"] = ("Volume", "sum")
    weekly = grouped.agg(**agg_kwargs)

    try:
        first_trading_day = grouped.apply(lambda g: g.index[0], include_groups=False)
    except TypeError:
        # older pandas without include_groups kwarg
        first_trading_day = grouped.apply(lambda g: g.index[0])
    weekly.index = pd.DatetimeIndex(first_trading_day.values)
    weekly.index.name = daily.index.name

    return weekly.dropna(subset=["Close"]).sort_index()


def resample_monthly(daily: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Monthly OHLCV via resampling, labeled at month start."""
    if daily is None or daily.empty:
        return daily
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in daily.columns:
        agg["Volume"] = "sum"
    return (
        daily.resample("MS")
        .agg(agg)
        .dropna(subset=["Close"])
    )


def derive_weekly_monthly(daily: pd.DataFrame) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Convenience: both derived timeframes in one call."""
    return resample_weekly(daily), resample_monthly(daily)
