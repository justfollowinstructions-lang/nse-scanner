#!/usr/bin/env python3
"""
NSE Long-Term Investment Scanner
==================================
Scans NSE stocks for long-term investment opportunities (1–4 year horizon).

Philosophy:
  - Primary analysis on MONTHLY charts (the "big picture")
  - Weekly chart for trend confirmation
  - Daily chart for entry timing
  - Strict multi-factor fundamental scoring
  - Valuation-aware (won't recommend obviously overpriced stocks)
  - Tracks previously scanned stocks across runs

Scoring (out of 100):
  Fundamental  : 40 pts  (growth, profitability, balance sheet, ownership)
  Technical    : 30 pts  (monthly pattern, weekly/daily confirmation, index)
  Valuation    : 20 pts  (PEG, PE vs sector, PB, FCF yield)
  Momentum     : 10 pts  (RS vs Nifty, 52-week position)

Modes:
  --scan       Full investment scan (run bi-weekly or monthly)
  --watchlist  Quick rescan of previously identified stocks
  --history    Print scan history summary
  --test       Test run on 50 large-caps

Usage:
  python scanner.py --scan
  python scanner.py --scan --min-score 50
  python scanner.py --watchlist
  python scanner.py --test
"""

import os, sys, json, time, sqlite3, argparse, logging
from datetime import date, datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# ═══════════════════════════════════════════════════════════════════════
# PATHS & LOGGING
# ═══════════════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "invest_cache.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR    = os.path.join(BASE_DIR, "logs")

for d in [OUTPUT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

_IST = timezone(timedelta(hours=5, minutes=30))
def _now():   return datetime.now(_IST)
def _today(): return _now().date()
def _ist(fmt="%Y-%m-%d %H:%M IST"): return _now().strftime(fmt)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(LOG_DIR, f"scanner_{_today()}.log"),
            encoding="utf-8"),
    ],
)
log = logging.getLogger("scanner")

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
MAX_WORKERS       = 4
MIN_MONTHLY_BARS  = 24    # Need at least 2 years of monthly data
MIN_MARKET_CAP_CR = 200   # ₹200 Cr minimum

# Minimum fundamental thresholds to even consider a stock
HARD_FILTERS = {
    "min_roe":              10.0,   # ROE > 10%
    "max_debt_equity":      2.0,    # D/E < 2
    "min_revenue_growth":   5.0,    # Revenue growth > 5% (TTM or 3yr)
    "max_pe":               120.0,  # PE < 120 (avoid extreme overvaluation)
}

# Sector PE reference (approximate FY25 Indian market PEs)
SECTOR_PE = {
    "Technology":           28,
    "Information Technology": 28,
    "Consumer Cyclical":    40,
    "Consumer Defensive":   45,
    "Healthcare":           35,
    "Industrials":          30,
    "Financial Services":   18,
    "Basic Materials":      18,
    "Energy":               15,
    "Real Estate":          30,
    "Utilities":            20,
    "Communication Services": 25,
    "default":              30,
}

# Index symbols (must match what data_updater stored)
NIFTY50_SYM  = "^NSEI"
NIFTYMID_SYM = "^NSEMDCP50"
VIX_SYM      = "^INDIAVIX"

# Sector index map
SECTOR_INDEX = {
    "Technology":           "^CNXIT",
    "Information Technology": "^CNXIT",
    "Healthcare":           "^CNXPHARMA",
    "Consumer Cyclical":    "^CNXCONSUM",
    "Consumer Defensive":   "^CNXFMCG",
    "Industrials":          "^CNXINFRA",
    "Basic Materials":      "^CNXMETAL",
    "Energy":               "^CNXENERGY",
    "Financial Services":   "^CNXPSUBANK",
    "Real Estate":          "^CNXREALTY",
}

# ═══════════════════════════════════════════════════════════════════════
# DATABASE ACCESS
# ═══════════════════════════════════════════════════════════════════════
_db_lock = Lock()
_db_con  = None

def _get_db() -> sqlite3.Connection:
    global _db_con
    with _db_lock:
        if _db_con is None:
            _db_con = sqlite3.connect(CACHE_PATH, check_same_thread=False)
            _db_con.execute("PRAGMA journal_mode=WAL")
            _db_con.execute("PRAGMA synchronous=NORMAL")
            _db_con.execute("PRAGMA cache_size=-131072")
            # Ensure scan tables exist (may be first run)
            _db_con.executescript("""
                CREATE TABLE IF NOT EXISTS scan_results (
                    scan_id TEXT NOT NULL, scan_date TEXT NOT NULL,
                    stock TEXT NOT NULL, score REAL, grade TEXT,
                    result_json TEXT,
                    PRIMARY KEY (scan_id, stock)
                );
                CREATE TABLE IF NOT EXISTS stock_history (
                    stock TEXT PRIMARY KEY,
                    first_seen_date TEXT, first_price REAL, first_score REAL,
                    last_seen_date TEXT,  last_price REAL,  last_score REAL,
                    times_scanned INTEGER DEFAULT 0,
                    pattern_history TEXT
                );
            """)
            _db_con.commit()
        return _db_con


def read_cache(stock: str, tf: str = "1mo", limit: int = 9999) -> pd.DataFrame | None:
    """Read OHLCV from price_cache."""
    try:
        con = _get_db()
        df = pd.read_sql(
            f"SELECT date,open,high,low,close,volume FROM price_cache "
            f"WHERE stock=? AND tf=? ORDER BY date DESC LIMIT {limit}",
            con, params=(stock, tf)
        )
        if len(df) < 2:
            return None
        df = df.iloc[::-1].reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.columns = ["Open","High","Low","Close","Volume"]
        df.index.name = None
        return df.astype(float)
    except Exception:
        return None


def read_fund(stock: str) -> dict:
    """Read fundamentals from cache. Returns {} if not found."""
    try:
        con = _get_db()
        row = con.execute(
            "SELECT fund_json FROM fund_cache WHERE stock=?", (stock,)
        ).fetchone()
        if row and row[0]:
            return json.loads(row[0])
    except Exception:
        pass
    return {}


def load_universe_from_cache() -> list[str]:
    """All stocks that have at least monthly OHLCV data."""
    try:
        con = _get_db()
        rows = con.execute(
            "SELECT DISTINCT stock FROM cache_meta WHERE tf='1mo'"
        ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════
# MARKET CONTEXT  (Nifty + VIX + breadth)
# ═══════════════════════════════════════════════════════════════════════
def load_market_context() -> dict:
    """
    Compute broad market state from Nifty50 monthly + VIX.
    Returns dict: regime, aggression (0-3), vix, trend_monthly, trend_weekly.
    """
    ctx = {
        "regime":         "Unknown",
        "aggression":     1,
        "vix":            None,
        "nifty_trend_mo": "Unknown",
        "nifty_trend_wk": "Unknown",
        "nifty_above_10mo_ma": False,
        "nifty_above_30mo_ma": False,
    }

    # Monthly Nifty
    mo = read_cache(NIFTY50_SYM, "1mo")
    if mo is not None and len(mo) >= 30:
        c = mo["Close"].values
        ctx["nifty_above_10mo_ma"] = bool(c[-1] > np.mean(c[-10:]))
        ctx["nifty_above_30mo_ma"] = bool(c[-1] > np.mean(c[-min(30, len(c)):]))
        ma10  = np.mean(c[-10:])
        ma30  = np.mean(c[-min(30, len(c)):])
        if c[-1] > ma10 > ma30:
            ctx["nifty_trend_mo"] = "Stage2-Bull"
        elif c[-1] > ma30:
            ctx["nifty_trend_mo"] = "Uptrend"
        elif c[-1] < ma10 < ma30:
            ctx["nifty_trend_mo"] = "Stage4-Bear"
        else:
            ctx["nifty_trend_mo"] = "Choppy"

    # Weekly Nifty
    wk = read_cache(NIFTY50_SYM, "1wk")
    if wk is not None and len(wk) >= 52:
        c = wk["Close"].values
        ma10 = np.mean(c[-10:]); ma40 = np.mean(c[-40:])
        if c[-1] > ma10 > ma40:
            ctx["nifty_trend_wk"] = "Uptrend"
        elif c[-1] > ma40:
            ctx["nifty_trend_wk"] = "Sideways"
        else:
            ctx["nifty_trend_wk"] = "Downtrend"

    # VIX
    vix_df = read_cache(VIX_SYM, "1mo")
    if vix_df is not None and len(vix_df) > 0:
        ctx["vix"] = round(float(vix_df["Close"].values[-1]), 2)

    # Overall regime
    mo_bull = ctx["nifty_trend_mo"] in ("Stage2-Bull", "Uptrend")
    wk_bull = ctx["nifty_trend_wk"] in ("Uptrend", "Sideways")
    vix_ok  = ctx["vix"] is None or ctx["vix"] < 22

    if mo_bull and wk_bull and vix_ok:
        ctx["regime"] = "Bull"
        ctx["aggression"] = 3
    elif mo_bull and vix_ok:
        ctx["regime"] = "Uptrend"
        ctx["aggression"] = 2
    elif mo_bull and not vix_ok:
        ctx["regime"] = "Cautious"
        ctx["aggression"] = 1
    elif not mo_bull and not wk_bull:
        ctx["regime"] = "Bear"
        ctx["aggression"] = 0
    else:
        ctx["regime"] = "Mixed"
        ctx["aggression"] = 1

    return ctx


def get_sector_trend(sector: str) -> str:
    """Return trend of the relevant sector index."""
    sym = SECTOR_INDEX.get(sector)
    if not sym:
        return "Unknown"
    df = read_cache(sym, "1mo")
    if df is None or len(df) < 12:
        return "Unknown"
    c = df["Close"].values
    ma10 = np.mean(c[-min(10, len(c)):])
    ma30 = np.mean(c[-min(30, len(c)):]) if len(c) >= 30 else ma10
    if c[-1] > ma10 > ma30:
        return "Stage2"
    if c[-1] > ma30:
        return "Uptrend"
    if c[-1] < ma10 < ma30:
        return "Downtrend"
    return "Choppy"


# ═══════════════════════════════════════════════════════════════════════
# MONTHLY PATTERN DETECTORS
# All detectors receive monthly close + volume arrays and return:
# {pattern, status, quality (0-1), bz (breakout zone), bottom,
#  last (current close), duration_months, details}
# or None if no pattern found.
# ═══════════════════════════════════════════════════════════════════════

def _vsurge_mo(v: np.ndarray, lb: int = 12) -> float | None:
    """Monthly volume surge vs 12-month average."""
    if v is None or len(v) < lb:
        return None
    avg = np.mean(v[-lb:])
    return round(float(v[-1] / avg), 2) if avg > 0 else None


def det_monthly_cup(c: np.ndarray, v: np.ndarray) -> dict | None:
    """
    Multi-year Cup & Handle on monthly chart.
    Cup: 12–48 months. Depth: 20–60%. Handle: 2–8 months.
    """
    n = len(c)
    if n < 24:
        return None

    s = pd.Series(c).rolling(3, min_periods=1).mean().values
    best = None

    for cup_len in range(12, min(49, n - 2)):
        seg = s[n - cup_len:]
        ti  = int(np.argmin(seg))
        if not (int(cup_len * 0.25) <= ti <= int(cup_len * 0.75)):
            continue

        lm = np.max(seg[:ti + 1])
        rm = np.max(seg[ti:])
        pk = max(lm, rm)
        tr = seg[ti]

        depth = (pk - tr) / pk
        if not (0.18 <= depth <= 0.65):
            continue

        sym = abs(lm - rm) / pk
        if sym > 0.25:
            continue

        # Handle portion: last 2-8 months should be a mild retracement
        handle_len = min(6, cup_len // 4)
        if handle_len < 2:
            continue
        handle = seg[-handle_len:]
        handle_drop = (np.max(handle) - np.min(handle)) / np.max(handle) if np.max(handle) > 0 else 1
        if handle_drop > 0.30:
            continue

        # Quality score
        try:
            x  = np.arange(ti + 1)
            cf = np.polyfit(x, seg[:ti + 1], 2)
            ssr = np.sum((seg[:ti + 1] - np.polyval(cf, x)) ** 2)
            sst = np.sum((seg[:ti + 1] - np.mean(seg[:ti + 1])) ** 2)
            r2  = 1 - ssr / sst if sst > 0 else 0
            if cf[0] <= 0 or r2 < 0.45:
                continue
        except Exception:
            continue

        q = r2 * (1 - sym) * (1 - handle_drop * 0.5)
        if best is None or q > best["q"]:
            best = dict(q=q, pk=pk, tr=tr, depth=depth, sym=sym,
                        cup_len=cup_len, handle_len=handle_len, r2=r2)

    if best is None:
        return None

    vs  = _vsurge_mo(v)
    bz  = round(float(best["pk"]), 2)
    bo  = c[-1] >= bz * 0.97 and (vs is not None and vs >= 1.3)
    return dict(
        pattern="MonthlyCupHandle",
        status="Breakout Ready" if bo else "Forming",
        quality=round(best["q"], 3),
        bz=bz, bottom=round(float(best["tr"]), 2),
        last=round(float(c[-1]), 2),
        duration_months=best["cup_len"],
        details=(f"Cup {best['cup_len']}mo | Depth {best['depth']*100:.0f}% | "
                 f"Sym {best['sym']*100:.0f}% | R² {best['r2']:.2f}"),
        vs=vs,
    )


def det_monthly_vcp(c: np.ndarray, v: np.ndarray) -> dict | None:
    """
    Volatility Contraction Pattern on monthly chart.
    Each contraction must be ≤80% of prior. Final depth ≤10%.
    """
    n = len(c)
    if n < 18:
        return None

    atr  = np.mean(np.abs(np.diff(c))) if n > 1 else np.mean(c) * 0.02
    prom = max(atr * 1.5, np.mean(c) * 0.015)

    try:
        highs, _ = find_peaks(c, prominence=prom, distance=3)
        lows,  _ = find_peaks(-c, prominence=prom, distance=3)
    except Exception:
        return None

    if len(highs) < 2 or len(lows) < 2:
        return None

    contractions = []
    hl = list(highs) + [n]
    for i, hi in enumerate(hl[:-1]):
        nh = hl[i + 1]
        nl = lows[(lows > hi) & (lows < nh)]
        lo = nl[0] if len(nl) > 0 else hi + int(np.argmin(c[hi:nh])) if nh - hi >= 3 else -1
        if lo < 0 or lo >= n:
            continue
        depth = (c[hi] - c[lo]) / c[hi]
        if depth < 0.04:
            continue
        contractions.append((hi, lo, depth))

    if len(contractions) < 3:
        return None

    depths = [ct[2] for ct in contractions]
    if not all(depths[i] <= depths[i - 1] * 0.85 for i in range(1, len(depths))):
        return None

    if contractions[-1][1] < int(n * 0.5):
        return None

    final_depth = depths[-1]
    if final_depth > 0.12:
        return None

    pivot = float(np.max(c[highs]))
    vs    = _vsurge_mo(v)
    bo    = c[-1] >= pivot * 0.97 and (vs is not None and vs >= 1.4)

    return dict(
        pattern="MonthlyVCP",
        status="Breakout Ready" if bo else "Forming",
        quality=round((1 - final_depth) * (len(contractions) / 5), 3),
        bz=round(pivot, 2),
        bottom=round(float(c[contractions[-1][1]]), 2),
        last=round(float(c[-1]), 2),
        duration_months=contractions[-1][1] - contractions[0][0],
        details=(f"{len(contractions)} contractions | "
                 f"First depth {depths[0]*100:.0f}% → Final {final_depth*100:.0f}%"),
        vs=vs,
    )


def det_monthly_flat_base(c: np.ndarray, v: np.ndarray) -> dict | None:
    """
    Multi-year Flat Base on monthly chart.
    6–24 months of sideways action with < 15% range.
    """
    n = len(c)
    if n < 12:
        return None

    best = None
    for bl in range(6, min(25, n) + 1):
        base = c[-bl:]
        bh   = np.max(base)
        blo  = np.min(base)
        rng  = (bh - blo) / bh if bh > 0 else 1
        if rng > 0.15:
            break   # Bases get wider as we go back — stop here

        # Prior trend must show at least 20% gain
        if n - bl >= 12:
            prior = c[n - bl - 12: n - bl]
            prior_gain = (c[n - bl] - prior[0]) / prior[0] if prior[0] > 0 else 0
        else:
            prior_gain = 0

        if prior_gain < 0.15:
            continue

        q = prior_gain * (1 - rng)
        if best is None or q > best["q"]:
            best = dict(q=q, bh=bh, blo=blo, rng=rng, bl=bl, prior_gain=prior_gain)

    if best is None:
        return None

    vs = _vsurge_mo(v)
    bz = round(float(best["bh"]), 2)
    bo = c[-1] >= bz * 0.98 and (vs is not None and vs >= 1.2)

    return dict(
        pattern="MonthlyFlatBase",
        status="Breakout Ready" if bo else "Forming",
        quality=round(best["q"], 3),
        bz=bz,
        bottom=round(float(best["blo"]), 2),
        last=round(float(c[-1]), 2),
        duration_months=best["bl"],
        details=(f"Base {best['bl']}mo | Range {best['rng']*100:.0f}% | "
                 f"Prior gain {best['prior_gain']*100:.0f}%"),
        vs=vs,
    )


def det_monthly_stage2(c: np.ndarray, v: np.ndarray) -> dict | None:
    """
    Stage 2 Breakout (Weinstein) on monthly chart.
    Stock crosses above declining 30-month MA (or was below for 6+ months).
    """
    n = len(c)
    if n < 30:
        return None

    ma30      = np.mean(c[-30:])
    ma30_prev = np.mean(c[-30 - 12:-12]) if n >= 42 else np.mean(c[-30:])
    ma10      = np.mean(c[-10:])
    ma10_prev = np.mean(c[-10 - 3:-3]) if n >= 13 else ma10

    # Must be above 30-month MA now
    if c[-1] <= ma30:
        return None

    # MA30 must be flattening or turning up (from a prior decline)
    ma_turning = ma30 >= ma30_prev * 0.97   # MA is at least flat

    # Must have been below MA within last 12 months
    recently_below = any(c[i] < np.mean(c[max(0, i - 30): i])
                         for i in range(n - 12, n - 1))

    if not recently_below and not (ma30 < ma30_prev * 0.98):
        return None   # Never below — not a stage change

    vs = _vsurge_mo(v)
    bo = vs is not None and vs >= 1.5

    return dict(
        pattern="MonthlyStage2",
        status="Breaking Out" if bo else "Stage Change",
        quality=round(min((c[-1] - ma30) / ma30, 0.30) * (1.2 if recently_below else 0.8), 3),
        bz=round(float(ma30), 2),
        bottom=round(float(np.min(c[-24:])), 2),
        last=round(float(c[-1]), 2),
        duration_months=24,
        details=(f"30MA={ma30:.0f} | Current {c[-1]:.0f} "
                 f"(+{(c[-1]-ma30)/ma30*100:.1f}%) | MA turning={'Yes' if ma_turning else 'No'}"),
        vs=vs,
    )


def det_monthly_asc_triangle(c: np.ndarray, v: np.ndarray) -> dict | None:
    """
    Multi-year Ascending Triangle on monthly chart.
    Flat resistance tested 2+ times, rising support trendline.
    """
    n = len(c)
    if n < 18:
        return None

    atr  = np.mean(np.abs(np.diff(c))) if n > 1 else np.mean(c) * 0.02
    prom = max(atr * 1.2, np.mean(c) * 0.01)

    try:
        pks, _ = find_peaks(c, prominence=prom, distance=3)
        trs, _ = find_peaks(-c, prominence=prom, distance=3)
    except Exception:
        return None

    if len(pks) < 2 or len(trs) < 2:
        return None

    pp  = c[pks]
    res = np.median(pp)
    if (np.max(pp) - np.min(pp)) / res > 0.05:
        return None   # Resistance not flat enough

    tp     = c[trs]
    slopes = [(tp[j] - tp[i]) / (trs[j] - trs[i])
              for i in range(len(trs)) for j in range(i + 1, len(trs))
              if trs[j] != trs[i]]
    if not slopes or np.median(slopes) <= 0:
        return None   # Support not rising

    rise = (tp[-1] - tp[0]) / tp[0] if tp[0] > 0 else 0
    if rise < 0.10 or trs[-1] < int(n * 0.4):
        return None

    vs = _vsurge_mo(v)
    bz = round(float(res), 2)
    bo = c[-1] >= bz * 0.98 and (vs is not None and vs >= 1.2)

    return dict(
        pattern="MonthlyAscTriangle",
        status="Breakout Ready" if bo else "Forming",
        quality=round(rise * (1 - (np.max(pp) - np.min(pp)) / res), 3),
        bz=bz,
        bottom=round(float(tp[0]), 2),
        last=round(float(c[-1]), 2),
        duration_months=int(trs[-1] - pks[0]),
        details=(f"{len(pks)} resistance tests | Support rise {rise*100:.0f}% | "
                 f"Resistance flat {(np.max(pp)-np.min(pp))/res*100:.1f}%"),
        vs=vs,
    )


def det_monthly_double_bottom(c: np.ndarray, v: np.ndarray) -> dict | None:
    """Multi-year W-bottom / Double Bottom on monthly chart."""
    n = len(c)
    if n < 24:
        return None

    try:
        troughs, _ = find_peaks(-c, prominence=0.05 * np.mean(c), distance=4)
    except Exception:
        return None

    if len(troughs) < 2:
        return None

    best = None
    for i in range(len(troughs) - 1):
        for j in range(i + 1, len(troughs)):
            sep = troughs[j] - troughs[i]
            if not (6 <= sep <= 48):
                continue
            p1, p2 = c[troughs[i]], c[troughs[j]]
            diff   = abs(p1 - p2) / min(p1, p2)
            if diff > 0.10:
                continue
            mid_max = np.max(c[troughs[i]:troughs[j] + 1])
            mr      = (mid_max - (p1 + p2) / 2) / ((p1 + p2) / 2)
            if mr < 0.08 or troughs[j] >= n - 2:
                continue
            sc = mr - diff
            if best is None or sc > best["sc"]:
                best = dict(sc=sc, mid=mid_max, diff=diff, mr=mr,
                            bottom=min(p1, p2), sep=sep, j=troughs[j])

    if best is None:
        return None

    vs = _vsurge_mo(v)
    bz = round(float(best["mid"]), 2)
    bo = c[-1] >= bz * 0.98 and (vs is not None and vs >= 1.2)

    return dict(
        pattern="MonthlyDoubleBottom",
        status="Breakout Ready" if bo else "Forming",
        quality=round(best["sc"], 3),
        bz=bz,
        bottom=round(float(best["bottom"]), 2),
        last=round(float(c[-1]), 2),
        duration_months=best["sep"],
        details=(f"Sep {best['sep']}mo | Mid rally {best['mr']*100:.0f}% | "
                 f"Bottom asymmetry {best['diff']*100:.0f}%"),
        vs=vs,
    )


# ═══════════════════════════════════════════════════════════════════════
# WEEKLY & DAILY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
def analyze_weekly(wk_df: pd.DataFrame | None) -> dict:
    """Weekly trend and volume expansion analysis."""
    result = {"trend": "Unknown", "above_40wma": False, "vol_expanding": False,
               "score": 0}
    if wk_df is None or len(wk_df) < 20:
        return result

    c = wk_df["Close"].values.astype(float)
    v = wk_df["Volume"].values.astype(float)
    n = len(c)

    ma10 = np.mean(c[-10:]) if n >= 10 else c[-1]
    ma40 = np.mean(c[-40:]) if n >= 40 else np.mean(c)

    result["above_40wma"] = bool(c[-1] > ma40)

    if c[-1] > ma10 > ma40:
        result["trend"] = "Uptrend"
        result["score"] = 8
    elif c[-1] > ma40:
        result["trend"] = "Sideways-Above-MA"
        result["score"] = 5
    elif c[-1] > ma10:
        result["trend"] = "Sideways"
        result["score"] = 3
    else:
        result["trend"] = "Downtrend"
        result["score"] = 0

    # Volume: last 4wk up-weeks vs prior 4wk up-weeks
    if n >= 20 and v is not None:
        up_weeks_recent = sum(1 for i in range(-4, 0) if c[i] > c[i - 1])
        up_weeks_prior  = sum(1 for i in range(-8, -4) if c[i] > c[i - 1])
        vol_up_weeks    = np.mean(v[-4:]) if len(v) >= 4 else 0
        vol_all         = np.mean(v[-20:]) if len(v) >= 20 else 0
        result["vol_expanding"] = bool(vol_up_weeks > vol_all * 1.1 and up_weeks_recent >= 3)

    return result


def analyze_daily(d_df: pd.DataFrame | None) -> dict:
    """Daily setup: tight consolidation above moving averages."""
    result = {"setup": "Unknown", "above_50dma": False, "tightening": False,
               "score": 0}
    if d_df is None or len(d_df) < 30:
        return result

    c = d_df["Close"].values.astype(float)
    v = d_df["Volume"].values.astype(float) if "Volume" in d_df.columns else None
    n = len(c)

    ma50  = np.mean(c[-50:])  if n >= 50  else np.mean(c)
    ma200 = np.mean(c[-200:]) if n >= 200 else np.mean(c)

    result["above_50dma"]  = bool(c[-1] > ma50)

    # Tightening: recent 10-day range < prior 20-day range
    if n >= 30:
        recent_range = (np.max(c[-10:]) - np.min(c[-10:])) / np.mean(c[-10:])
        prior_range  = (np.max(c[-30:-10]) - np.min(c[-30:-10])) / np.mean(c[-30:-10])
        result["tightening"] = bool(recent_range < prior_range * 0.7)

    if c[-1] > ma50 > ma200:
        if result["tightening"]:
            result["setup"] = "Tight-Above-MAs"
            result["score"] = 4
        else:
            result["setup"] = "Uptrend"
            result["score"] = 2
    elif c[-1] > ma200:
        result["setup"] = "Constructive"
        result["score"] = 1
    else:
        result["setup"] = "Below-MAs"
        result["score"] = 0

    return result


# ═══════════════════════════════════════════════════════════════════════
# RELATIVE STRENGTH
# ═══════════════════════════════════════════════════════════════════════
def calc_rs_vs_nifty(stock_monthly: pd.DataFrame, n50_monthly: pd.DataFrame,
                     years: int = 1) -> float | None:
    """Stock return vs Nifty50 over N years (monthly data)."""
    months = years * 12
    try:
        # Align on common dates
        combined = pd.DataFrame({
            "stock":  stock_monthly["Close"],
            "nifty":  n50_monthly["Close"],
        }).dropna()
        if len(combined) < months:
            return None
        sr = (combined["stock"].iloc[-1] / combined["stock"].iloc[-months] - 1) * 100
        nr = (combined["nifty"].iloc[-1] / combined["nifty"].iloc[-months] - 1) * 100
        return round(sr - nr, 2)
    except Exception:
        return None


def calc_rs_percentile(rs_1yr: float | None, universe_rs: list[float]) -> float | None:
    """Return percentile of this stock's RS vs universe."""
    if rs_1yr is None or not universe_rs:
        return None
    try:
        from scipy.stats import percentileofscore
        return round(percentileofscore(universe_rs, rs_1yr), 1)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# SCORING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def score_fundamentals(fund: dict) -> tuple[float, list[str]]:
    """
    Score fundamentals 0–40 points.
    Returns (score, reasons_list).
    """
    score   = 0.0
    reasons = []

    # ── Revenue Growth (0-8 pts) ─────────────────────────────────────────────
    rev_growth = fund.get("sales_3yr_cagr") or fund.get("revenue_growth_ttm")
    if rev_growth is not None:
        if rev_growth > 25:   score += 8; reasons.append(f"Revenue CAGR {rev_growth:.0f}%")
        elif rev_growth > 20: score += 6; reasons.append(f"Revenue CAGR {rev_growth:.0f}%")
        elif rev_growth > 15: score += 4; reasons.append(f"Revenue CAGR {rev_growth:.0f}%")
        elif rev_growth > 10: score += 2

    # ── Profit Growth (0-8 pts) ──────────────────────────────────────────────
    profit_growth = fund.get("profit_3yr_cagr") or fund.get("earnings_growth_ttm")
    if profit_growth is not None:
        if profit_growth > 30:   score += 8; reasons.append(f"Profit CAGR {profit_growth:.0f}%")
        elif profit_growth > 25: score += 6; reasons.append(f"Profit CAGR {profit_growth:.0f}%")
        elif profit_growth > 20: score += 4; reasons.append(f"Profit CAGR {profit_growth:.0f}%")
        elif profit_growth > 15: score += 2

    # ── ROE (0-6 pts) ────────────────────────────────────────────────────────
    roe = fund.get("roe_screener") or fund.get("roe")
    if roe is not None:
        if roe > 25:   score += 6; reasons.append(f"ROE {roe:.0f}%")
        elif roe > 20: score += 4; reasons.append(f"ROE {roe:.0f}%")
        elif roe > 15: score += 2
        elif roe > 12: score += 1

    # ── ROCE (0-6 pts) ───────────────────────────────────────────────────────
    roce = fund.get("roce")
    if roce is not None:
        if roce > 25:   score += 6; reasons.append(f"ROCE {roce:.0f}%")
        elif roce > 20: score += 4; reasons.append(f"ROCE {roce:.0f}%")
        elif roce > 15: score += 2
        elif roce > 12: score += 1

    # ── Debt Management (0-6 pts) ────────────────────────────────────────────
    de = fund.get("debt_to_equity_ratio") or fund.get("debt_equity")
    if de is not None:
        # Normalize: if it looks like it's in % form (e.g. 45 meaning 0.45)
        if de > 10:
            de = de / 100
        if de < 0.1:    score += 6; reasons.append("Nearly debt-free")
        elif de < 0.3:  score += 5; reasons.append(f"Low D/E {de:.2f}")
        elif de < 0.5:  score += 4
        elif de < 1.0:  score += 2
        elif de < 1.5:  score += 0
        else:           score -= 2  # Penalize high debt

    # ── Promoter Holding (0-4 pts) ───────────────────────────────────────────
    ph    = fund.get("promoter_holding")
    pledge = fund.get("promoter_pledge", 0) or 0
    if ph is not None:
        if ph > 60 and pledge < 5:    score += 4; reasons.append(f"Promoter {ph:.0f}% holding")
        elif ph > 50 and pledge < 10: score += 3; reasons.append(f"Promoter {ph:.0f}% holding")
        elif ph > 40 and pledge < 20: score += 2
        elif pledge > 30:             score -= 2  # Heavy pledge is a red flag

    # ── Operating Margin (0-4 pts) ───────────────────────────────────────────
    opm = fund.get("operating_margin_screener") or fund.get("operating_margin")
    if opm is not None:
        if opm > 25:   score += 4; reasons.append(f"OPM {opm:.0f}%")
        elif opm > 20: score += 3
        elif opm > 15: score += 2
        elif opm > 10: score += 1

    # ── Free Cash Flow (0-2 pts) ─────────────────────────────────────────────
    fcf = fund.get("free_cashflow")
    if fcf is not None:
        if fcf > 0:
            score += 2; reasons.append("Positive FCF")

    return round(max(0, min(score, 40)), 2), reasons


def score_technical(pattern_result: dict | None, wk: dict, d: dict,
                    market_ctx: dict, sector_trend: str) -> tuple[float, list[str]]:
    """Score technical setup 0–30 points."""
    score   = 0.0
    reasons = []

    # ── Monthly Pattern (0-15 pts) ───────────────────────────────────────────
    if pattern_result:
        q   = pattern_result.get("quality", 0)
        status = pattern_result.get("status", "")
        pat    = pattern_result.get("pattern", "")
        dur    = pattern_result.get("duration_months", 0)

        # Base score from quality
        base = round(q * 12, 2)

        # Breakout bonus
        if "Breaking Out" in status or "Breakout Ready" in status:
            base = min(15, base + 3)
            reasons.append(f"{pat} — {status}")
        else:
            reasons.append(f"{pat} forming ({dur}mo base)")

        # Duration bonus — longer bases = bigger moves
        if dur >= 24:  base = min(15, base + 1.5)
        elif dur >= 18: base = min(15, base + 1.0)
        elif dur >= 12: base = min(15, base + 0.5)

        score += base

    # ── Weekly Confirmation (0-8 pts) ────────────────────────────────────────
    wk_score = wk.get("score", 0)
    score += wk_score
    if wk["trend"] == "Uptrend":
        reasons.append("Weekly uptrend")
    if wk.get("vol_expanding"):
        reasons.append("Weekly volume expanding")

    # ── Daily Setup (0-4 pts) ────────────────────────────────────────────────
    d_score = d.get("score", 0)
    score += d_score
    if d["setup"] == "Tight-Above-MAs":
        reasons.append("Tight daily consolidation above MAs")

    # ── Index / Sector Alignment (0-3 pts) ───────────────────────────────────
    regime = market_ctx.get("regime", "Unknown")
    if regime in ("Bull", "Strong-Bull"):
        score += 3; reasons.append("Bull market")
    elif regime in ("Uptrend", "Cautious"):
        score += 2
    elif regime == "Bear":
        score = max(0, score - 2)

    if sector_trend == "Stage2":
        score = min(30, score + 1); reasons.append("Sector in Stage2")

    return round(min(score, 30), 2), reasons


def score_valuation(fund: dict, cmp: float) -> tuple[float, list[str]]:
    """Score valuation 0–20 points."""
    score   = 0.0
    reasons = []

    pe = fund.get("pe_ratio") or fund.get("pe_screener")
    pb = fund.get("pb_ratio")

    # ── PEG Ratio (0-8 pts) ──────────────────────────────────────────────────
    growth = fund.get("profit_3yr_cagr") or fund.get("earnings_growth_ttm")
    if pe and pe > 0 and growth and growth > 0:
        peg = pe / growth
        fund["_peg"] = round(peg, 2)
        if peg < 0.5:   score += 8; reasons.append(f"PEG {peg:.2f} — deeply undervalued")
        elif peg < 1.0: score += 6; reasons.append(f"PEG {peg:.2f} — undervalued")
        elif peg < 1.5: score += 4; reasons.append(f"PEG {peg:.2f} — fair value")
        elif peg < 2.0: score += 2
        # PEG > 2 = no bonus (overvalued relative to growth)

    # ── PE vs Sector (0-4 pts) ───────────────────────────────────────────────
    sector = fund.get("sector", "")
    sector_pe = SECTOR_PE.get(sector, SECTOR_PE["default"])
    if pe and pe > 0:
        pe_ratio = pe / sector_pe
        if pe_ratio < 0.70:   score += 4; reasons.append(f"PE {pe:.0f} below sector avg ({sector_pe})")
        elif pe_ratio < 0.90: score += 3; reasons.append(f"PE {pe:.0f} vs sector {sector_pe}")
        elif pe_ratio < 1.10: score += 2
        elif pe_ratio < 1.50: score += 0
        else:                  score -= 1   # Expensive vs peers

    # ── PB Ratio (0-4 pts) ───────────────────────────────────────────────────
    # For high-quality businesses, higher PB is OK if ROE is high
    roe = fund.get("roe_screener") or fund.get("roe") or 0
    if pb and pb > 0:
        # Justified PB = ROE / Cost_of_equity. For 15% COE: JPB = ROE/15
        justified_pb = max(1.0, (roe or 15) / 15) if roe else 2.0
        if pb < justified_pb * 0.7:   score += 4; reasons.append(f"PB {pb:.1f} below justified value")
        elif pb < justified_pb:       score += 2
        elif pb < justified_pb * 1.5: score += 1

    # ── Analyst Target Upside (0-4 pts) ──────────────────────────────────────
    tgt = fund.get("analyst_target")
    if tgt and cmp and cmp > 0:
        upside = (tgt - cmp) / cmp * 100
        if upside > 40:   score += 4; reasons.append(f"Analyst target ₹{tgt:.0f} (+{upside:.0f}%)")
        elif upside > 25: score += 3; reasons.append(f"Analyst target ₹{tgt:.0f} (+{upside:.0f}%)")
        elif upside > 15: score += 2
        elif upside > 0:  score += 1

    return round(min(score, 20), 2), reasons


def score_momentum(mo_df: pd.DataFrame | None, n50_mo: pd.DataFrame | None,
                   fund: dict) -> tuple[float, list[str]]:
    """Score momentum / relative strength 0–10 points."""
    score   = 0.0
    reasons = []

    # ── RS vs Nifty 1yr (0-6 pts) ────────────────────────────────────────────
    rs_1yr = None
    if mo_df is not None and n50_mo is not None:
        rs_1yr = calc_rs_vs_nifty(mo_df, n50_mo, years=1)
    if rs_1yr is not None:
        if rs_1yr > 30:   score += 6; reasons.append(f"Outperforms Nifty by {rs_1yr:.0f}%/yr")
        elif rs_1yr > 20: score += 4; reasons.append(f"Outperforms Nifty by {rs_1yr:.0f}%/yr")
        elif rs_1yr > 10: score += 2
        elif rs_1yr > 0:  score += 1

    # ── 52-Week Position (0-4 pts) ───────────────────────────────────────────
    wkh = fund.get("fifty_two_week_high")
    cmp = fund.get("regular_market_price")
    if wkh and cmp and wkh > 0:
        dist = (wkh - cmp) / wkh * 100
        if dist < 5:    score += 4; reasons.append("Near 52-week high")
        elif dist < 15: score += 2
        elif dist < 25: score += 1

    return round(min(score, 10), 2), reasons


# ═══════════════════════════════════════════════════════════════════════
# TARGET CALCULATION  (fundamental-driven for long-term)
# ═══════════════════════════════════════════════════════════════════════
def calc_lt_targets(cmp: float, fund: dict,
                    pattern: dict | None) -> dict:
    """
    Compute 1yr / 2yr / 3yr price targets for long-term investing.

    Method:
    1. Estimate forward EPS using current EPS + growth rate
    2. Apply a sustainable terminal PE (based on quality)
    3. Cross-check with analyst consensus and pattern measured move
    """
    result = {
        "target_1yr": None, "target_2yr": None, "target_3yr": None,
        "fair_value":  None, "stop_loss": None,
        "expected_cagr_2yr": None, "upside_potential_pct": None,
    }

    if cmp <= 0:
        return result

    # ── Stop Loss ────────────────────────────────────────────────────────────
    # For investing: stop at 15% below entry OR below pattern bottom
    if pattern and pattern.get("bottom") and pattern["bottom"] > 0:
        pattern_stop = pattern["bottom"] * 0.95
        result["stop_loss"] = round(max(cmp * 0.85, pattern_stop), 2)
    else:
        result["stop_loss"] = round(cmp * 0.85, 2)

    # ── Fair Value (Graham / fundamental-based) ───────────────────────────────
    pe        = fund.get("pe_ratio") or fund.get("pe_screener")
    growth    = fund.get("profit_3yr_cagr") or fund.get("earnings_growth_ttm") or 15
    roe       = fund.get("roe_screener") or fund.get("roe") or 15
    sector    = fund.get("sector", "")
    sector_pe = SECTOR_PE.get(sector, SECTOR_PE["default"])

    # Sustainable PE = min(sector_PE, 25 + growth/2) for quality businesses
    sustainable_pe = min(sector_pe * 1.2, 25 + max(0, growth) / 2)
    sustainable_pe = max(12, min(sustainable_pe, 60))

    if pe and pe > 0 and growth > 0:
        # Earnings grow at `growth`% pa; revalue at sustainable PE in 2yr
        earnings_yield_now = 1 / pe   # E/P
        fwd_eps_2yr  = (1 / pe) * ((1 + growth / 100) ** 2)   # earnings grow
        fair_value   = round(cmp * fwd_eps_2yr * sustainable_pe, 2)
        result["fair_value"] = fair_value
        result["target_2yr"] = fair_value
        result["target_1yr"] = round(cmp * (1 + growth / 200), 2)   # half the 2yr move
        result["target_3yr"] = round(cmp * ((1 + growth / 100) ** 2.5) * (sustainable_pe / max(pe, 10)), 2)

    # ── Override with analyst target if available and sane ────────────────────
    analyst_tgt = fund.get("analyst_target")
    if analyst_tgt and analyst_tgt > cmp * 0.8:
        # Blend: 60% fundamental, 40% analyst
        if result["target_2yr"]:
            result["target_2yr"] = round(result["target_2yr"] * 0.6 + analyst_tgt * 0.4, 2)
        else:
            result["target_2yr"] = round(analyst_tgt, 2)

    # ── Pattern measured move (cross-check) ──────────────────────────────────
    if pattern and pattern.get("bz") and pattern.get("bottom"):
        bz     = pattern["bz"]
        bottom = pattern["bottom"]
        depth  = bz - bottom
        pat_target = bz + depth   # classic measured move
        if result["target_2yr"]:
            # Take higher of the two (conservative: capped at 2× measured move)
            pat_t2 = round(min(pat_target, bz + depth * 2), 2)
            result["target_2yr"] = max(result["target_2yr"], pat_t2)
        else:
            result["target_2yr"] = round(pat_target, 2)

    # ── Derived metrics ───────────────────────────────────────────────────────
    if result["target_2yr"] and result["target_2yr"] > cmp:
        result["upside_potential_pct"] = round((result["target_2yr"] - cmp) / cmp * 100, 1)
        result["expected_cagr_2yr"]    = round(((result["target_2yr"] / cmp) ** 0.5 - 1) * 100, 1)

    return result


# ═══════════════════════════════════════════════════════════════════════
# WHY BUY / KEY RISKS
# ═══════════════════════════════════════════════════════════════════════
def build_why_buy(fund_reasons: list, tech_reasons: list,
                  val_reasons: list, mom_reasons: list,
                  fund: dict, targets: dict) -> str:
    """Assemble the top investment thesis in one line."""
    all_reasons = fund_reasons + val_reasons + tech_reasons + mom_reasons
    # De-duplicate while preserving order
    seen = set()
    unique = []
    for r in all_reasons:
        if r not in seen:
            seen.add(r); unique.append(r)

    # Add CAGR expectation
    cagr = targets.get("expected_cagr_2yr")
    if cagr and cagr > 0:
        unique.insert(0, f"Expected ~{cagr:.0f}% CAGR (2yr)")

    return " | ".join(unique[:6])


def build_key_risks(fund: dict, pattern: dict | None) -> str:
    """List the top 2-3 risks to flag."""
    risks = []

    de    = fund.get("debt_to_equity_ratio") or fund.get("debt_equity")
    if de and de > 1.0:
        risks.append(f"High D/E {de:.1f}")

    pledge = fund.get("promoter_pledge", 0) or 0
    if pledge > 20:
        risks.append(f"Promoter pledge {pledge:.0f}%")

    pe = fund.get("pe_ratio") or fund.get("pe_screener")
    if pe and pe > 50:
        risks.append(f"High PE {pe:.0f}")

    growth = fund.get("profit_3yr_cagr") or fund.get("earnings_growth_ttm")
    if growth is not None and growth < 10:
        risks.append("Slow profit growth")

    ne = fund.get("next_earnings")
    if ne:
        risks.append(f"Earnings due {ne}")

    if pattern and pattern.get("status") == "Forming":
        risks.append("Pattern still forming (no breakout yet)")

    return " | ".join(risks[:4]) if risks else "None identified"


# ═══════════════════════════════════════════════════════════════════════
# HARD FILTER
# ═══════════════════════════════════════════════════════════════════════
def passes_hard_filter(fund: dict, cmp: float) -> tuple[bool, str]:
    """
    Apply minimum criteria. Returns (pass: bool, reason: str).
    Stocks failing these are not scored at all.
    """
    # Market cap
    mc = fund.get("market_cap")
    if mc:
        mc_cr = mc / 1e7
        if mc_cr < MIN_MARKET_CAP_CR:
            return False, f"Market cap ₹{mc_cr:.0f}Cr < minimum"

    # ROE
    roe = fund.get("roe_screener") or fund.get("roe")
    if roe is not None and roe < HARD_FILTERS["min_roe"]:
        return False, f"ROE {roe:.1f}% below minimum {HARD_FILTERS['min_roe']}%"

    # D/E
    de = fund.get("debt_to_equity_ratio") or fund.get("debt_equity")
    if de is not None:
        de_norm = de / 100 if de > 10 else de
        if de_norm > HARD_FILTERS["max_debt_equity"]:
            return False, f"D/E {de_norm:.2f} above maximum"

    # Revenue growth
    rev = fund.get("sales_3yr_cagr") or fund.get("revenue_growth_ttm")
    if rev is not None and rev < HARD_FILTERS["min_revenue_growth"]:
        return False, f"Revenue growth {rev:.1f}% below minimum"

    # PE cap
    pe = fund.get("pe_ratio") or fund.get("pe_screener")
    if pe is not None and pe > HARD_FILTERS["max_pe"]:
        return False, f"PE {pe:.0f} above maximum {HARD_FILTERS['max_pe']}"

    return True, "OK"


# ═══════════════════════════════════════════════════════════════════════
# SCAN ONE STOCK
# ═══════════════════════════════════════════════════════════════════════
def scan_one(sym: str, n50_mo: pd.DataFrame | None,
             market_ctx: dict) -> dict | None:
    """
    Full investment analysis for one stock.
    Returns a result dict or None if stock doesn't qualify.
    """
    # ── Load data ────────────────────────────────────────────────────────────
    mo_df = read_cache(sym, "1mo")
    wk_df = read_cache(sym, "1wk")
    d_df  = read_cache(sym, "1d")
    fund  = read_fund(sym)

    # Need at minimum: monthly data + some fundamental info
    if mo_df is None or len(mo_df) < MIN_MONTHLY_BARS:
        return None
    if not fund:
        return None

    c = mo_df["Close"].values.astype(float)
    v = mo_df["Volume"].values.astype(float) if "Volume" in mo_df.columns else None
    cmp = round(float(c[-1]), 2)

    # ── Hard filter ───────────────────────────────────────────────────────────
    passes, reason = passes_hard_filter(fund, cmp)
    if not passes:
        log.debug(f"{sym}: filtered — {reason}")
        return None

    # ── Monthly pattern detection (try all 6, keep best) ────────────────────
    detectors = [
        det_monthly_cup,
        det_monthly_vcp,
        det_monthly_flat_base,
        det_monthly_stage2,
        det_monthly_asc_triangle,
        det_monthly_double_bottom,
    ]

    patterns_found = []
    for det in detectors:
        try:
            res = det(c, v)
            if res is not None:
                patterns_found.append(res)
        except Exception:
            pass

    # Best pattern by quality
    best_pattern = max(patterns_found, key=lambda p: p["quality"]) if patterns_found else None

    # Convergence: multiple patterns = higher conviction
    converging = "+".join(sorted(set(p["pattern"] for p in patterns_found))) \
        if len(patterns_found) > 1 else None

    # ── Sector trend ─────────────────────────────────────────────────────────
    sector_trend = get_sector_trend(fund.get("sector", ""))

    # ── Multi-timeframe analysis ──────────────────────────────────────────────
    wk_analysis = analyze_weekly(wk_df)
    d_analysis  = analyze_daily(d_df)

    # ── Scoring ───────────────────────────────────────────────────────────────
    f_score, f_reasons = score_fundamentals(fund)
    t_score, t_reasons = score_technical(best_pattern, wk_analysis, d_analysis,
                                         market_ctx, sector_trend)
    v_score, v_reasons = score_valuation(fund, cmp)
    m_score, m_reasons = score_momentum(mo_df, n50_mo, fund)

    total = round(f_score + t_score + v_score + m_score, 1)

    # ── Rating grade ──────────────────────────────────────────────────────────
    if total >= 80:   grade = "STRONG BUY ⭐⭐⭐⭐⭐"
    elif total >= 70: grade = "BUY ⭐⭐⭐⭐"
    elif total >= 60: grade = "ACCUMULATE ⭐⭐⭐"
    elif total >= 50: grade = "WATCH ⭐⭐"
    elif total >= 35: grade = "MONITOR ⭐"
    else:             grade = "SKIP"

    # ── Targets ───────────────────────────────────────────────────────────────
    targets = calc_lt_targets(cmp, fund, best_pattern)

    # ── Investment horizon & risk ─────────────────────────────────────────────
    horizon = "2-3 years"
    risk    = "Medium"
    de      = fund.get("debt_to_equity_ratio") or fund.get("debt_equity") or 0
    if de and de > 10:
        de = de / 100
    if total >= 75 and (de or 0) < 0.3:
        horizon = "1-3 years"; risk = "Low-Medium"
    elif total >= 65:
        horizon = "2-3 years"; risk = "Medium"
    elif total < 55:
        horizon = "3-4 years"; risk = "Medium-High"

    # ── Market cap classification ─────────────────────────────────────────────
    mc     = fund.get("market_cap") or 0
    mc_cr  = mc / 1e7
    if mc_cr >= 20000:  cap_class = "Large-Cap"
    elif mc_cr >= 5000: cap_class = "Mid-Cap"
    elif mc_cr >= 500:  cap_class = "Small-Cap"
    else:               cap_class = "Micro-Cap"

    # ── RS metrics ────────────────────────────────────────────────────────────
    rs_1yr = calc_rs_vs_nifty(mo_df, n50_mo, years=1) if n50_mo is not None else None
    rs_3yr = calc_rs_vs_nifty(mo_df, n50_mo, years=3) if n50_mo is not None else None

    # ── Why Buy / Key Risks ───────────────────────────────────────────────────
    why_buy   = build_why_buy(f_reasons, t_reasons, v_reasons, m_reasons, fund, targets)
    key_risks = build_key_risks(fund, best_pattern)

    # ── Data quality ──────────────────────────────────────────────────────────
    sources = fund.get("_sources", [])
    data_fields = [
        fund.get("roe"), fund.get("roce"), fund.get("debt_to_equity_ratio"),
        fund.get("promoter_holding"), fund.get("sales_3yr_cagr"),
        fund.get("profit_3yr_cagr"), fund.get("pe_ratio"), fund.get("pb_ratio"),
        fund.get("operating_margin"),
    ]
    data_quality = round(sum(1 for f in data_fields if f is not None) / len(data_fields) * 100)

    # ── 52-week metrics ───────────────────────────────────────────────────────
    wkh = fund.get("fifty_two_week_high")
    wkl = fund.get("fifty_two_week_low")
    dist_52h = round((wkh - cmp) / wkh * 100, 1) if wkh and wkh > 0 else None
    dist_52l = round((cmp - wkl) / wkl * 100, 1) if wkl and wkl > 0 else None

    # ── Volume surge monthly ──────────────────────────────────────────────────
    vs_mo = _vsurge_mo(v) if v is not None else None

    return {
        # Identification
        "Stock":                    sym.replace(".NS", ""),
        "Company_Name":             fund.get("long_name", ""),
        "Sector":                   fund.get("sector", ""),
        "Industry":                 fund.get("industry", ""),
        "Market_Cap_Cr":            round(mc_cr, 0),
        "Cap_Class":                cap_class,

        # Monthly Pattern
        "Monthly_Pattern":          best_pattern["pattern"] if best_pattern else "None",
        "Monthly_Pattern_Status":   best_pattern["status"]  if best_pattern else "",
        "Monthly_Pattern_Quality":  best_pattern["quality"] if best_pattern else 0,
        "Monthly_Pattern_Duration_Months": best_pattern["duration_months"] if best_pattern else 0,
        "Monthly_Pattern_Details":  best_pattern["details"] if best_pattern else "",
        "Converging_Signals":       converging or "",

        # Multi-timeframe
        "Weekly_Trend":             wk_analysis["trend"],
        "Weekly_Above_40MA":        str(wk_analysis["above_40wma"]),
        "Weekly_Volume_Expanding":  str(wk_analysis.get("vol_expanding", False)),
        "Daily_Setup":              d_analysis["setup"],
        "Daily_Above_50MA":         str(d_analysis["above_50dma"]),
        "Sector_Trend":             sector_trend,
        "Index_Regime":             market_ctx.get("regime", "Unknown"),
        "Nifty_Monthly_Trend":      market_ctx.get("nifty_trend_mo", ""),

        # Price Levels
        "CMP":                      cmp,
        "Breakout_Zone":            best_pattern["bz"]     if best_pattern else "",
        "Pattern_Bottom":           best_pattern["bottom"] if best_pattern else "",
        "Stop_Loss":                targets["stop_loss"],
        "Target_1yr":               targets["target_1yr"],
        "Target_2yr":               targets["target_2yr"],
        "Target_3yr":               targets["target_3yr"],
        "Fair_Value_Estimate":      targets["fair_value"],
        "Upside_Potential_Pct":     targets["upside_potential_pct"],
        "Expected_CAGR_2yr_Pct":    targets["expected_cagr_2yr"],

        # Valuation
        "PE_Ratio":                 _fmt(fund.get("pe_ratio") or fund.get("pe_screener")),
        "Forward_PE":               _fmt(fund.get("forward_pe")),
        "PB_Ratio":                 _fmt(fund.get("pb_ratio")),
        "PEG_Ratio":                _fmt(fund.get("_peg")),
        "EV_EBITDA":                _fmt(fund.get("ev_ebitda")),
        "Dividend_Yield_Pct":       _fmt(fund.get("dividend_yield") or fund.get("dividend_yield_screener")),
        "Analyst_Target":           _fmt(fund.get("analyst_target")),
        "Analyst_Count":            fund.get("analyst_count", ""),

        # Profitability
        "ROE_Pct":                  _fmt(fund.get("roe_screener") or fund.get("roe")),
        "ROCE_Pct":                 _fmt(fund.get("roce")),
        "Operating_Margin_Pct":     _fmt(fund.get("operating_margin_screener") or fund.get("operating_margin")),
        "Net_Margin_Pct":           _fmt(fund.get("net_margin")),
        "Return_on_Assets_Pct":     _fmt(fund.get("roa")),
        "Gross_Margin_Pct":         _fmt(fund.get("gross_margin")),

        # Growth
        "Revenue_Growth_TTM_Pct":   _fmt(fund.get("sales_growth_ttm") or fund.get("revenue_growth_ttm")),
        "Revenue_Growth_3yr_CAGR":  _fmt(fund.get("sales_3yr_cagr")),
        "Profit_Growth_TTM_Pct":    _fmt(fund.get("profit_growth_ttm") or fund.get("earnings_growth_ttm")),
        "Profit_Growth_3yr_CAGR":   _fmt(fund.get("profit_3yr_cagr")),
        "EPS_Growth_YoY_Pct":       _fmt(fund.get("earnings_quarterly_growth")),

        # Balance Sheet
        "Debt_to_Equity":           _fmt(fund.get("debt_to_equity_ratio") or fund.get("debt_equity")),
        "Current_Ratio":            _fmt(fund.get("current_ratio")),
        "Interest_Coverage":        _fmt(fund.get("interest_coverage")),
        "Free_Cashflow_Cr":         _fmt(_to_cr(fund.get("free_cashflow"))),

        # Ownership
        "Promoter_Holding_Pct":     _fmt(fund.get("promoter_holding")),
        "Promoter_Pledge_Pct":      _fmt(fund.get("promoter_pledge")),
        "FII_Holding_Pct":          _fmt(fund.get("fii_holding") or fund.get("held_pct_institutions")),
        "DII_Holding_Pct":          _fmt(fund.get("dii_holding")),
        "Insider_Holding_Pct":      _fmt(fund.get("held_pct_insiders")),

        # Price History
        "52wk_High":                _fmt(wkh),
        "52wk_Low":                 _fmt(wkl),
        "Dist_From_52wk_High_Pct":  dist_52h,
        "Dist_From_52wk_Low_Pct":   dist_52l,
        "Beta":                     _fmt(fund.get("beta")),

        # Momentum / RS
        "RS_vs_Nifty_1yr_Pct":      _fmt(rs_1yr),
        "RS_vs_Nifty_3yr_Pct":      _fmt(rs_3yr),
        "Monthly_Vol_Surge":        _fmt(vs_mo),

        # Scores
        "Fundamental_Score_40":     f_score,
        "Technical_Score_30":       t_score,
        "Valuation_Score_20":       v_score,
        "Momentum_Score_10":        m_score,
        "Overall_Score_100":        total,
        "Rating_Grade":             grade,

        # Analysis
        "Why_Buy":                  why_buy,
        "Key_Risks":                key_risks,
        "Investment_Horizon":       horizon,
        "Risk_Level":               risk,

        # Metadata
        "Data_Sources":             ", ".join(sources) if sources else "yfinance",
        "Data_Quality_Pct":         data_quality,
        "Monthly_Bars_Available":   len(mo_df),
        "Next_Earnings":            fund.get("next_earnings", ""),
        "Scan_Date":                str(_today()),
    }


def _fmt(v) -> str | float:
    """Format numeric values for CSV — blank if None."""
    if v is None:
        return ""
    try:
        f = float(v)
        if f != f:   # NaN
            return ""
        return round(f, 2)
    except Exception:
        return str(v)


def _to_cr(val) -> float | None:
    """Convert INR value to Cr (÷1e7)."""
    if val is None:
        return None
    try:
        return round(float(val) / 1e7, 2)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# SCAN HISTORY
# ═══════════════════════════════════════════════════════════════════════
def update_history(results: list[dict]):
    """Update stock_history table with latest scan results."""
    con = _get_db()
    with _db_lock:
        for r in results:
            stock = r["Stock"]
            price = r["CMP"]
            score = r["Overall_Score_100"]

            existing = con.execute(
                "SELECT first_seen_date, first_price, first_score, times_scanned, pattern_history "
                "FROM stock_history WHERE stock=?", (stock,)
            ).fetchone()

            if existing:
                pats = json.loads(existing[4] or "[]")
                pat  = r.get("Monthly_Pattern", "")
                if pat and pat not in pats:
                    pats.append(pat)
                con.execute(
                    "UPDATE stock_history SET last_seen_date=?, last_price=?, last_score=?, "
                    "times_scanned=times_scanned+1, pattern_history=? WHERE stock=?",
                    (str(_today()), price, score, json.dumps(pats), stock)
                )
            else:
                con.execute(
                    "INSERT INTO stock_history (stock,first_seen_date,first_price,first_score,"
                    "last_seen_date,last_price,last_score,times_scanned,pattern_history) "
                    "VALUES (?,?,?,?,?,?,?,1,?)",
                    (stock, str(_today()), price, score,
                     str(_today()), price, score,
                     json.dumps([r.get("Monthly_Pattern", "")]))
                )
        con.commit()


def load_history() -> dict:
    """Load stock_history as {stock: row_dict}."""
    try:
        con = _get_db()
        rows = con.execute("SELECT * FROM stock_history").fetchall()
        cols = [d[0] for d in con.execute("SELECT * FROM stock_history LIMIT 0").description]
        return {r[0]: dict(zip(cols, r)) for r in rows}
    except Exception:
        return {}


def enrich_with_history(results: list[dict], history: dict) -> list[dict]:
    """Add first-seen date and price-change-since-first-seen columns."""
    for r in results:
        stock = r["Stock"]
        h = history.get(stock)
        if h:
            r["First_Seen_Date"]              = h.get("first_seen_date", "")
            r["Days_On_Watchlist"]            = (
                (_today() - datetime.strptime(h["first_seen_date"], "%Y-%m-%d").date()).days
                if h.get("first_seen_date") else ""
            )
            fp = h.get("first_price")
            cmp = r["CMP"]
            r["Price_Change_Since_First_Seen_Pct"] = (
                round((cmp - fp) / fp * 100, 1) if fp and fp > 0 else ""
            )
            r["Times_Scanned"]                = h.get("times_scanned", 1)
            fs = h.get("first_score")
            r["Score_vs_First_Seen"]          = (
                round(r["Overall_Score_100"] - fs, 1) if fs else ""
            )
        else:
            r["First_Seen_Date"]              = str(_today())
            r["Days_On_Watchlist"]            = 0
            r["Price_Change_Since_First_Seen_Pct"] = ""
            r["Times_Scanned"]                = 1
            r["Score_vs_First_Seen"]          = ""
    return results


# ═══════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════
def save_results(results: list[dict], scan_id: str):
    """Save results to DB and CSV files."""
    if not results:
        log.warning("No results to save")
        return

    # DB
    con = _get_db()
    with _db_lock:
        for r in results:
            con.execute(
                "INSERT OR REPLACE INTO scan_results (scan_id,scan_date,stock,score,grade,result_json) "
                "VALUES (?,?,?,?,?,?)",
                (scan_id, str(_today()), r["Stock"],
                 r["Overall_Score_100"], r["Rating_Grade"], json.dumps(r))
            )
        con.commit()

    # CSV
    ts   = _now().strftime("%Y%m%d_%H%M")
    df   = pd.DataFrame(results)

    # All results
    all_csv = os.path.join(OUTPUT_DIR, f"invest_scan_ALL_{ts}.csv")
    df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    # Strong Buy + Buy only
    buy_mask = df["Rating_Grade"].str.startswith("STRONG BUY") | df["Rating_Grade"].str.startswith("BUY")
    buy_df   = df[buy_mask]
    buy_csv  = os.path.join(OUTPUT_DIR, f"invest_scan_BUY_{ts}.csv")
    buy_df.to_csv(buy_csv, index=False, encoding="utf-8-sig")

    # Accumulate + Watch
    watch_mask = df["Rating_Grade"].str.startswith("ACCUMULATE") | df["Rating_Grade"].str.startswith("WATCH")
    watch_df   = df[watch_mask]
    watch_csv  = os.path.join(OUTPUT_DIR, f"invest_scan_WATCH_{ts}.csv")
    watch_df.to_csv(watch_csv, index=False, encoding="utf-8-sig")

    log.info(f"\n{'='*60}")
    log.info(f"SCAN COMPLETE — {len(results)} stocks rated")
    log.info(f"  STRONG BUY + BUY : {len(buy_df)}")
    log.info(f"  ACCUMULATE+WATCH : {len(watch_df)}")
    log.info(f"  ALL CSV  → {os.path.basename(all_csv)}")
    log.info(f"  BUY CSV  → {os.path.basename(buy_csv)}")
    log.info(f"  WATCH CSV→ {os.path.basename(watch_csv)}")
    log.info(f"{'='*60}")

    if len(buy_df) > 0:
        print("\n─── TOP BUY CANDIDATES ───")
        show_cols = ["Stock","Cap_Class","Monthly_Pattern","CMP",
                     "Target_2yr","Expected_CAGR_2yr_Pct","Overall_Score_100",
                     "Rating_Grade","Why_Buy"]
        show = buy_df[[c for c in show_cols if c in buy_df.columns]].head(20)
        print(show.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════
# MAIN SCAN RUNNERS
# ═══════════════════════════════════════════════════════════════════════
def run_full_scan(stocks: list[str], min_score: float = 35.0) -> list[dict]:
    """Run full investment scan on all stocks."""
    log.info(f"=== INVESTMENT SCAN: {len(stocks)} stocks | min_score={min_score} ===")
    t0 = time.time()

    # Market context
    mkt = load_market_context()
    log.info(f"Market: {mkt['regime']} | Nifty monthly: {mkt['nifty_trend_mo']} | "
             f"VIX: {mkt.get('vix', 'N/A')}")

    # Nifty50 monthly for RS calculation
    n50_mo = read_cache(NIFTY50_SYM, "1mo")
    if n50_mo is None:
        log.warning("Nifty50 monthly data not found — RS vs Nifty won't be computed")

    # Parallel scan
    all_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(scan_one, s, n50_mo, mkt): s for s in stocks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            if done % 100 == 0:
                log.info(f"  {done}/{len(stocks)} scanned | {len(all_results)} qualified")
            try:
                r = fut.result()
                if r and r["Overall_Score_100"] >= min_score:
                    all_results.append(r)
            except Exception as e:
                pass

    # Sort by score desc
    all_results.sort(key=lambda x: -x["Overall_Score_100"])

    elapsed = time.time() - t0
    log.info(f"Scan done: {elapsed:.0f}s | {len(all_results)} stocks scored ≥ {min_score}")
    return all_results


def run_watchlist_scan(min_score: float = 35.0) -> list[dict]:
    """Rescan only previously seen stocks (fast check)."""
    history = load_history()
    if not history:
        log.warning("No history found — run --scan first")
        return []

    stocks = [s + ".NS" for s in history.keys()]
    log.info(f"Watchlist rescan: {len(stocks)} previously identified stocks")

    mkt   = load_market_context()
    n50_mo = read_cache(NIFTY50_SYM, "1mo")

    results = []
    for sym in stocks:
        try:
            r = scan_one(sym, n50_mo, mkt)
            if r:
                results.append(r)
        except Exception:
            pass

    results.sort(key=lambda x: -x["Overall_Score_100"])
    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="NSE Long-Term Investment Scanner")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scan",      action="store_true",
                      help="Full scan of all stocks")
    mode.add_argument("--watchlist", action="store_true",
                      help="Quick rescan of previously identified stocks")
    mode.add_argument("--history",   action="store_true",
                      help="Print scan history summary")
    mode.add_argument("--test",      action="store_true",
                      help="Test run on ~50 large-caps")

    ap.add_argument("--min-score",   type=float, default=35.0,
                    help="Minimum score to include in results (default 35)")
    ap.add_argument("--limit",       type=int,   default=0,
                    help="Limit number of stocks (for testing)")
    args = ap.parse_args()

    # Ensure DB exists
    _get_db()

    # ── History ────────────────────────────────────────────────────────────────
    if args.history:
        hist = load_history()
        if not hist:
            print("No history yet. Run --scan first.")
            return
        print(f"\n{'Stock':<15} {'First Seen':<12} {'First ₹':>10} "
              f"{'Last ₹':>10} {'Δ%':>7} {'Score':>7} {'Scanned':>8}")
        print("-" * 75)
        rows = sorted(hist.values(), key=lambda x: x.get("last_score", 0), reverse=True)
        for r in rows:
            fp  = r.get("first_price", 0) or 0
            lp  = r.get("last_price",  0) or 0
            chg = round((lp - fp) / fp * 100, 1) if fp > 0 else 0
            print(f"{r['stock']:<15} {r.get('first_seen_date',''):<12} "
                  f"₹{fp:>9.2f} ₹{lp:>9.2f} {chg:>+7.1f}% "
                  f"{r.get('last_score',0):>7.1f} {r.get('times_scanned',0):>8}")
        return

    # ── Test mode ──────────────────────────────────────────────────────────────
    if args.test:
        TEST_STOCKS = [
            "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC","SBIN",
            "BHARTIARTL","KOTAKBANK","LT","AXISBANK","BAJFINANCE","ASIANPAINT","MARUTI",
            "SUNPHARMA","TITAN","WIPRO","ULTRACEMCO","BAJAJFINSV","NESTLEIND","POWERGRID",
            "NTPC","TECHM","HCLTECH","JSWSTEEL","TATASTEEL","ADANIENT","ADANIPORTS",
            "ONGC","COALINDIA","BRITANNIA","DIVISLAB","DRREDDY","EICHERMOT","GRASIM",
            "HDFCLIFE","INDUSINDBK","BAJAJ-AUTO","CIPLA","APOLLOHOSP","HAVELLS",
            "HEROMOTOCO","LUPIN","MARICO","PIDILITIND","SIEMENS","TRENT","DIXON",
        ]
        stocks = [s + ".NS" for s in TEST_STOCKS]
        log.info(f"Test mode: {len(stocks)} large-cap stocks")

        scan_id = f"test_{_today()}"
        mkt    = load_market_context()
        n50_mo = read_cache(NIFTY50_SYM, "1mo")
        history = load_history()

        results = []
        for sym in stocks:
            try:
                r = scan_one(sym, n50_mo, mkt)
                if r:
                    results.append(r)
            except Exception as e:
                log.debug(f"{sym}: {e}")

        results.sort(key=lambda x: -x["Overall_Score_100"])
        results = enrich_with_history(results, history)
        update_history(results)
        save_results(results, scan_id)
        return

    # ── Watchlist rescan ───────────────────────────────────────────────────────
    if args.watchlist:
        history = load_history()
        results = run_watchlist_scan(args.min_score)
        results = enrich_with_history(results, history)
        update_history(results)
        scan_id = f"watchlist_{_today()}"
        save_results(results, scan_id)
        return

    # ── Full scan ──────────────────────────────────────────────────────────────
    if args.scan:
        stocks = load_universe_from_cache()
        if not stocks:
            log.error("No stocks in cache. Run: python data_updater.py --bootstrap first")
            sys.exit(1)

        if args.limit > 0:
            stocks = stocks[:args.limit]
            log.info(f"Limited to {len(stocks)} stocks")

        history = load_history()
        results = run_full_scan(stocks, args.min_score)
        results = enrich_with_history(results, history)
        update_history(results)
        scan_id = f"scan_{_today()}"
        save_results(results, scan_id)


if __name__ == "__main__":
    main()
