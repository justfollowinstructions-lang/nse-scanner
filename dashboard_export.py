#!/usr/bin/env python3
"""
Dashboard Export — self-contained interactive HTML dashboard
================================================================
Ported from the cup_test scanner's chart_export.py + chart_viewer/
template.html — same self-contained-file / vendored-JS / template-
substitution mechanism, generalized from a single Cup&Handle detector
to nse-scanner's 14 pattern detectors.

Design (unchanged from cup_test):
  - One flat list of "symbol cards", one per (stock, pattern, timeframe)
    signal, tagged with which of 7 categories it belongs to (this is
    signal_tags.py's tag system, not cup_test's Cup&Handle-derived
    categories). A signal in multiple categories (e.g. a Very-High-
    Quality pick that's also in today's signals) is represented once
    with all its category memberships — this drives the sidebar's "×2"
    multi-category badge.
  - Every card carries pre-computed explanation text (via explain.py)
    and OHLCV data for all three timeframes so the chart, timeframe
    switcher, and comparisons all work client-side with no further
    data fetching once the HTML is open.
  - template.html placeholders are substituted directly — no Jinja
    dependency, matches cup_test exactly.

Called once from scanner.py's main(), after CSV/watchlist/Telegram
output — reads the same tagged DataFrame + watchlist + signal_outcomes
data those already used, so nothing is a second source of truth.
"""

from __future__ import annotations

import json
import math
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd

import explain
import signal_tags
import resample_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(BASE_DIR, "chart_viewer", "template.html")
LWC_JS_PATH = os.path.join(BASE_DIR, "vendor", "lightweight-charts.standalone.production.js")

CHART_LOOKBACK_DAILY_BARS = 200      # ~10 months — plenty of context for
                                      # swing-pattern review; trimmed from
                                      # 300 to help keep total file size
                                      # browser-friendly at nse-scanner's
                                      # signal volume (see MAX_TOTAL_CARDS)
CHART_LOOKBACK_WEEKLY_BARS = 156     # 3 years
CHART_LOOKBACK_MONTHLY_BARS = 96     # 8 years

# Hard ceiling on how many signals get a full interactive chart card
# embedded in the HTML. See _apply_card_budget()'s docstring for why —
# short version: nse-scanner can produce 6,000+ signals/day, and
# embedding OHLCV for all of them produces a 150MB+ file no browser can
# open. Every signal is still in the CSVs regardless of this cap.
MAX_TOTAL_CARDS = 800

# signal_tags tag key -> template.html CATEGORY_DEFS key
_TAG_TO_CATEGORY = {
    "HIGH_CONVICTION": "confirmed",
    "BUY_STRONG":      "verge",
    "BUY_MODERATE":    "watchlist",
    "WATCH":           "early_watch",
}


def export_html_dashboard(
    df_today: pd.DataFrame,
    market_trend: str,
    scan_date: str,
    read_cache_fn: Callable[[str, str, int], Optional[pd.DataFrame]],
    con=None,
    output_dir: Optional[str] = None,
    log=None,
) -> Optional[str]:
    """
    Main entry point. Returns the output path on success, None if there
    was nothing to export or template/JS assets are missing (logged as
    a warning — never raises, so it can't break the scan run).

    df_today        : today's signals DataFrame (already has score10/
                       tier/tags columns from signal_tags.tag_dataframe)
    market_trend     : e.g. "Bullish" / "Bearish" / "Neutral"
    read_cache_fn    : read_cache(stock, tf, limit) -> DataFrame|None,
                       nse-scanner's existing multi-TF cache reader
    con              : sqlite3 connection to signals.db, used to pull
                       Active Tracking / Historical rows from
                       signal_outcomes. If None, those two categories
                       are simply empty (dashboard still exports).
    """
    log = log or _NullLog()
    if not os.path.exists(TEMPLATE_PATH):
        log.warning(f"dashboard_export: template.html not found at {TEMPLATE_PATH} — skipping")
        return None
    if not os.path.exists(LWC_JS_PATH):
        log.warning(f"dashboard_export: lightweight-charts JS not found at {LWC_JS_PATH} — skipping")
        return None

    rows_by_category, merged = _collect_all_signals(df_today, con, scan_date, log)
    if not merged:
        log.info("dashboard_export: no signals to export today — skipping")
        return None

    ohlcv_cache: dict = {}
    symbol_cards = []
    skipped_no_cache = 0
    skipped_errors: list[str] = []
    for key, row in merged.items():
        try:
            card = _build_symbol_card(row, read_cache_fn, ohlcv_cache)
            if card is not None:
                symbol_cards.append(card)
            else:
                skipped_no_cache += 1
        except Exception as e:
            # Was log.debug (invisible at default INFO level) — this is
            # exactly the class of bug that made the ".NS" suffix
            # mismatch undiagnosable from logs alone. Now surfaced as a
            # WARNING (first 5 in full, rest counted) so a systematic
            # failure is never silent again.
            if len(skipped_errors) < 5:
                log.warning(f"dashboard_export: skipped {key} due to error: {e}")
            skipped_errors.append(str(key))

    if skipped_no_cache:
        log.info(f"dashboard_export: {skipped_no_cache} signals had no OHLCV cache data "
                 f"(stock not yet cached, or a symbol-format mismatch)")
    if skipped_errors:
        log.warning(f"dashboard_export: {len(skipped_errors)} signals skipped due to "
                    f"errors while building cards (see WARNING lines above for the first 5)")

    if not symbol_cards:
        log.info("dashboard_export: no symbol cards had chart data — skipping")
        return None

    data_payload = {
        "scan_date": scan_date,
        "nifty_trend": market_trend,
        "symbols": symbol_cards,
    }

    output_dir = output_dir or os.path.join(BASE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = _render_html(data_payload, scan_date, market_trend, output_dir)
    log.info(f"HTML dashboard written: {output_path} ({len(symbol_cards)} symbol cards)")
    return output_path


# ─── Collect + merge signals across categories ─────────────────────────────

def _collect_all_signals(df_today: pd.DataFrame, con, scan_date: str, log) -> tuple[dict, dict]:
    """
    Returns (rows_by_category, merged) where merged is keyed by
    (stock, pattern) -> row dict, with a "_categories" list attached
    (which of confirmed/verge/watchlist/today/early_watch/active/
    historical this signal belongs to) and "_is_high_conviction" for
    explain.py's rating.

    SCALE FIX: nse-scanner's 14 detectors can produce 6,000+ signals in
    a single day (vs. cup_test's single Cup&Handle detector, which this
    mechanism was originally sized for) — embedding full OHLCV chart
    data for all of them produced a 150MB+ HTML file, unopenable in a
    browser. Two changes: (1) AVOID-tier rows (score10 too low to earn
    ANY tag — signal_tags.compute_tags() returns []) are dropped
    entirely rather than flowing into "today" unconditionally; on a
    real 6,675-signal day these alone were ~2,700 of the ~4,100
    candidate cards. (2) what remains is capped to MAX_TOTAL_CARDS,
    prioritized so HIGH_CONVICTION/BUY_STRONG/Active/Historical are
    NEVER dropped (they're always small in number), BUY_MODERATE is
    kept in full up to the cap, and WATCH-tier fills any remaining
    budget ordered by score10. Every signal — capped or not — is still
    in the CSVs; this only limits what gets a full interactive chart.
    """
    rows_by_category: dict[str, list[dict]] = {k: [] for k in
        ["confirmed", "verge", "watchlist", "today", "early_watch", "active", "historical"]}
    merged: dict[tuple, dict] = {}
    categories_seen: dict[tuple, set] = {}

    def _add(row: dict, cat: str):
        pk = (row.get("stock"), row.get("pattern"))
        categories_seen.setdefault(pk, set()).add(cat)
        rows_by_category[cat].append(row)
        if pk not in merged:
            merged[pk] = row

    if df_today is not None and len(df_today):
        for _, r in df_today.iterrows():
            row = r.to_dict()
            row["_is_high_conviction"] = signal_tags.is_high_conviction(row)
            tags = str(row.get("tags") or "").split(",")
            matched_cats = [cat_key for tag_key, cat_key in _TAG_TO_CATEGORY.items()
                             if tag_key in tags]
            if not matched_cats:
                # AVOID tier (or otherwise untagged) — no card for this
                # one; still fully present in the CSVs, just not worth
                # a slot in a size-capped interactive dashboard.
                continue
            _add(row, "today")
            for cat_key in matched_cats:
                _add(row, cat_key)

    if con is not None:
        try:
            active_df = pd.read_sql_query(
                """SELECT s.*, so.entry_price AS oc_entry, so.return_5d AS oc_return_5d,
                          so.exit_type AS oc_exit_type
                   FROM signal_outcomes so
                   JOIN signals s ON s.stock=so.stock AND s.pattern=so.pattern
                        AND s.scan_date=so.signal_date
                   WHERE so.exit_type IS NULL OR so.exit_type='OPEN'
                   ORDER BY so.signal_date DESC LIMIT 200""",
                con,
            )
            for _, r in active_df.iterrows():
                row = r.to_dict()
                row["_is_high_conviction"] = False
                _add(row, "active")
        except Exception as e:
            log.debug(f"dashboard_export: active-tracking query failed: {e}")

        try:
            hist_df = pd.read_sql_query(
                """SELECT s.*, so.entry_price AS oc_entry, so.return_5d AS oc_return_5d,
                          so.exit_type AS oc_exit_type
                   FROM signal_outcomes so
                   JOIN signals s ON s.stock=so.stock AND s.pattern=so.pattern
                        AND s.scan_date=so.signal_date
                   WHERE so.exit_type IS NOT NULL AND so.exit_type != 'OPEN'
                   ORDER BY so.signal_date DESC LIMIT 200""",
                con,
            )
            for _, r in hist_df.iterrows():
                row = r.to_dict()
                row["_is_high_conviction"] = False
                _add(row, "historical")
        except Exception as e:
            log.debug(f"dashboard_export: historical query failed: {e}")

    for pk, row in merged.items():
        row["_categories"] = sorted(categories_seen.get(pk, set()))

    merged = _apply_card_budget(merged, log)
    return rows_by_category, merged


def _apply_card_budget(merged: dict, log) -> dict:
    """Priority-order and cap `merged` to MAX_TOTAL_CARDS. Always keeps
    active/historical and HIGH_CONVICTION/BUY_STRONG entries (small in
    number); trims the long tail of WATCH-tier entries first."""
    if len(merged) <= MAX_TOTAL_CARDS:
        return merged

    def _priority(row: dict) -> tuple:
        cats = set(row.get("_categories", []))
        never_drop = bool(cats & {"active", "historical", "confirmed", "verge"})
        is_moderate = "watchlist" in cats
        try:
            score = float(row.get("score10") or 0)
        except (TypeError, ValueError):
            score = 0.0
        # Sort descending on all three: never-drop first, then moderate
        # tier, then by score within each bucket.
        return (0 if never_drop else (1 if is_moderate else 2), -score)

    ordered = sorted(merged.items(), key=lambda kv: _priority(kv[1]))
    kept = dict(ordered[:MAX_TOTAL_CARDS])
    log.info(f"dashboard_export: {len(merged)} candidate signals, capped to "
             f"{len(kept)} full-chart cards (HIGH_CONVICTION/BUY_STRONG/BUY_MODERATE/"
             f"Active/Historical kept in full, WATCH-tier trimmed by score10 — "
             f"every signal is still in the CSVs)")
    return kept


# ─── Build one symbol card ─────────────────────────────────────────────────

def _build_symbol_card(row: dict, read_cache_fn, ohlcv_cache: dict) -> Optional[dict]:
    stock = row.get("stock")
    if not stock:
        return None

    # BUG FIX: scanner.py deliberately strips ".NS" before writing the
    # "stock" field into the signals table (stock=sym.replace(".NS","")
    # in scan_stock(), for clean CSV/Telegram display) — but price_cache
    # is keyed by the full yfinance ticker WITH the suffix (e.g.
    # "ASKAUTOLTD.NS"), since that's what batch_downloader/warm_cache
    # actually download under. Looking up read_cache_fn(stock, ...) with
    # the unsuffixed "stock" value was therefore a guaranteed miss for
    # every single row — this is why every symbol card was silently
    # failing ("no symbol cards had chart data"). Reconstruct the cache
    # key; keep the unsuffixed `stock` for display (card["symbol"]) so
    # it still matches what the CSV/Telegram output shows.
    cache_symbol = stock if stock.upper().endswith((".NS", ".BO")) else f"{stock}.NS"

    daily = ohlcv_cache.get(cache_symbol)
    if daily is None:
        daily = read_cache_fn(cache_symbol, "1d", 400)
        ohlcv_cache[cache_symbol] = daily
    if daily is None or daily.empty:
        return None

    timeframes_json = _build_timeframes_json(cache_symbol, daily, read_cache_fn)
    if not any(timeframes_json.values()):
        return None

    d = _derive_display_fields(row)
    rating = explain.overall_rating(row)
    reasons = explain.scanner_reasons(row)
    wb = explain.why_buy(row)
    fexp = explain.full_explanation(row)
    sell_notes = explain.sell_rules_text(row)

    card = {
        "symbol": stock,
        "company_name": row.get("name") or "",
        "sector": row.get("sector") or "",
        "timeframe": row.get("timeframe") or "daily",
        "categories": row.get("_categories", []),

        "pattern_type": row.get("pattern") or "",
        "pattern_stage": row.get("status") or "",
        "signal_type": row.get("status") or "",
        "tier_label": row.get("tier") or "",
        "score10": _num(row.get("score10")),
        "quality_score": d["quality_score"],
        "canslim_pct": d["canslim_pct"],

        "current_price": _num(row.get("cmp")),
        "entry_price": _num(row.get("breakout_zone")) or _num(row.get("cmp")),
        "pivot_point": _num(row.get("breakout_zone")),
        "stop_loss_price": _num(row.get("stop_loss")),
        "stop_loss_pct": d["stop_loss_pct"],
        "target1": _num(row.get("target_1")),
        "target2": _num(row.get("target_2")),
        "target3": _num(row.get("target_3")),
        "rr_t2": _num(row.get("risk_reward")),

        "breakout_readiness_pct": d["breakout_readiness_pct"],
        "price_vs_pivot_pct": d["price_vs_pivot_pct"],

        "volume_ratio": _num(row.get("vol_surge")),
        "volume_confirmed_label": d["volume_confirmed_label"],
        "rs_rating": _num(row.get("rs_percentile")),
        "rs_tag": d["rs_tag"],
        "prior_uptrend_tag": d["stage_label"],
        "readiness_above_50ma": "Stage2" in str(row.get("stage") or ""),

        "dist_52wk_pct": _num(row.get("dist_52wk_pct")),
        "formation_days": _int(row.get("formation_days")),
        "t1_eta": row.get("t1_eta") or "",
        "t2_eta": row.get("t2_eta") or "",
        "t3_eta": row.get("t3_eta") or "",
        "piotroski_score": _int(row.get("piotroski_score")),
        "pledge_pct": _num(row.get("pledge_pct")),
        "bulk_deal_cr": _num(row.get("bulk_deal_cr")),
        "mtf_confluence": row.get("converging") or "",

        "position_size_shares": None,
        "capital_required": None,

        "pattern_start_date": _datestr(row.get("pattern_start_date")),
        "pattern_end_date": _datestr(row.get("pattern_end_date")),

        "sell_notes": sell_notes,
        "status": row.get("status") or "",

        "rating": rating,
        "scanner_reasons": reasons,
        "why_buy": wb,
        "explanation": fexp,

        "timeframes": timeframes_json,
    }
    return card


def _derive_display_fields(row: dict) -> dict:
    def f(key, default=None):
        v = row.get(key)
        if v is None:
            return default
        try:
            fv = float(v)
            return default if math.isnan(fv) else fv
        except (TypeError, ValueError):
            return default

    score10 = f("score10", 0.0) or 0.0
    quality_score = round(score10 * 10, 1)

    canslim = f("canslim_score")
    dc = f("data_completeness")
    canslim_pct = round(100 * canslim / dc, 0) if (canslim is not None and dc) else None

    cmp_ = f("cmp")
    bz = f("breakout_zone")
    if cmp_ and bz:
        pct = round((cmp_ - bz) / bz * 100, 2)
        price_vs_pivot_pct = pct
        breakout_readiness_pct = max(0.0, 100.0 - min(abs(pct) * 10, 100.0))
    else:
        price_vs_pivot_pct = None
        breakout_readiness_pct = None

    stop = f("stop_loss")
    stop_loss_pct = round((cmp_ - stop) / cmp_ * 100, 2) if (cmp_ and stop) else None

    vs = f("vol_surge")
    volume_confirmed_label = "Yes" if (vs is not None and vs >= 1.4) else ("No" if vs is not None else None)

    rs = f("rs_percentile")
    if rs is None:
        rs_tag = None
    elif rs >= 90:
        rs_tag = "Leader"
    elif rs >= 70:
        rs_tag = "Strong"
    elif rs >= 50:
        rs_tag = "Average"
    else:
        rs_tag = "Weak"

    stage = str(row.get("stage") or "")
    stage_label = {
        "Stage2": "Stage 2 — Advancing",
        "Stage1": "Stage 1 — Basing",
        "Stage3": "Stage 3 — Topping",
        "Stage4": "Stage 4 — Declining",
    }.get(stage, stage or None)

    return {
        "quality_score": quality_score,
        "canslim_pct": canslim_pct,
        "price_vs_pivot_pct": price_vs_pivot_pct,
        "breakout_readiness_pct": breakout_readiness_pct,
        "stop_loss_pct": stop_loss_pct,
        "volume_confirmed_label": volume_confirmed_label,
        "rs_tag": rs_tag,
        "stage_label": stage_label,
    }


# ─── OHLCV -> JSON bars for all three timeframes ───────────────────────────

def _build_timeframes_json(stock: str, daily: pd.DataFrame, read_cache_fn) -> dict:
    out = {}
    try:
        out["1D"] = _bars_to_json(daily.tail(CHART_LOOKBACK_DAILY_BARS))
    except Exception:
        out["1D"] = []

    weekly = None
    try:
        weekly = read_cache_fn(stock, "1wk", CHART_LOOKBACK_WEEKLY_BARS)
    except Exception:
        weekly = None
    if weekly is None or weekly.empty:
        weekly = resample_utils.resample_weekly(daily)
    try:
        out["1W"] = _bars_to_json(weekly.tail(CHART_LOOKBACK_WEEKLY_BARS))
    except Exception:
        out["1W"] = []

    monthly = None
    try:
        monthly = read_cache_fn(stock, "1mo", CHART_LOOKBACK_MONTHLY_BARS)
    except Exception:
        monthly = None
    if monthly is None or monthly.empty:
        monthly = resample_utils.resample_monthly(daily)
    try:
        out["1M"] = _bars_to_json(monthly.tail(CHART_LOOKBACK_MONTHLY_BARS))
    except Exception:
        out["1M"] = []

    return out


def _bars_to_json(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    out = []
    for idx, row in df.iterrows():
        try:
            o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
            v = float(row["Volume"]) if "Volume" in row and pd.notna(row["Volume"]) else 0.0
            if any(math.isnan(x) for x in (o, h, l, c)):
                continue
            out.append({
                "time": pd.Timestamp(idx).strftime("%Y-%m-%d"),
                "o": round(o, 2), "h": round(h, 2), "l": round(l, 2), "c": round(c, 2),
                "v": round(v, 0),
            })
        except Exception:
            continue
    return out


# ─── Small safe-conversion helpers (NaN-safe, JSON-safe) ──────────────────

def _num(val, default=None):
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, 4)
    except (TypeError, ValueError):
        return default


def _int(val, default=None):
    n = _num(val, default)
    return int(n) if n is not None else default


def _datestr(val) -> Optional[str]:
    if val is None or val == "":
        return None
    try:
        return pd.Timestamp(val).strftime("%Y-%m-%d")
    except Exception:
        return str(val)


def _json_safe(obj):
    """Recursively replace NaN/Inf (incl. numpy scalar types) with None
    so json.dumps(..., allow_nan=False) never raises mid-export."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ─── Render: substitute placeholders into template.html ───────────────────

def _render_html(data_payload: dict, scan_date: str, nifty_trend: str, output_dir: str) -> str:
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = f.read()
    with open(LWC_JS_PATH, encoding="utf-8") as f:
        lwc_js = f.read()

    safe_payload = _json_safe(data_payload)
    data_json = json.dumps(safe_payload, ensure_ascii=False, allow_nan=False)

    html = template
    html = html.replace("__SCAN_DATE__", scan_date)
    html = html.replace("__NIFTY_TREND__", nifty_trend or "Unknown")
    html = html.replace("__SYMBOL_COUNT__", str(len(data_payload["symbols"])))
    html = html.replace("/*__LIGHTWEIGHT_CHARTS_JS__*/", lwc_js)
    html = html.replace("/*__CHART_DATA_JSON__*/", data_json)

    output_path = os.path.join(output_dir, f"nse_scanner_dashboard_{scan_date}.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Stable "latest" copy for convenience (bookmarkable / always-current link)
    latest_path = os.path.join(output_dir, "nse_scanner_dashboard_latest.html")
    try:
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception:
        pass

    return output_path


class _NullLog:
    def info(self, *a, **k): pass
    def debug(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
