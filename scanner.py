#!/usr/bin/env python3
"""
NSE Long-Term Investment Scanner  (v2 — fully audited)
=========================================================
All 52 audit issues addressed. Change log vs v1:

BUG-01  _vsurge_mo excludes current bar from baseline (v[-lb-1:-1])
BUG-02  3yr target uses exponent 3.0 (was 2.5)
BUG-03  1yr target uses proper forward EPS × sustainable PE formula
BUG-04  VCP quality capped at 1.0
BUG-05  VCP contraction threshold tightened to 0.80 (was 0.85)
BUG-06  Flat base loop uses continue not break — checks all lengths
BUG-07  Stage2 quality normalised to 0-1 scale like other patterns
BUG-08  Ascending triangle resistance tolerance raised to 10% (was 5%)
BUG-09  "Strong-Bull" dead code removed — regime check uses "Bull" only
BUG-13  SECTOR_INDEX now includes Utilities / Communication / Consumer Defensive
MIN-01  _fmt and _to_cr defined above scan_one (not after)
MIN-02  Targeted warnings suppression only
MIN-03  Per-thread read connections via threading.local()
SIG-01  Penalty applied for negative / near-zero profit growth
SIG-02  PE vs Sector replaced with EV/EBITDA vs sector (no double-counting)
SIG-03  Sustainable PE raised for Consumer Defensive / FMCG names
SIG-04  Stage2 uses EMA (10-period) not SMA for MA crossover
SIG-05  Cup, VCP, Ascending Triangle detectors use High/Low arrays for peaks/troughs
SIG-06  FCF scored as yield tiers (0-6 pts) not binary (was 0-2 pts)
SIG-07  OPM trend penalty added (-2 pts if margin declining)
SIG-08  Analyst target gated on minimum 3 analysts
SIG-10  Cup handle verifies prior uptrend >= 20% before the cup
SIG-11  Interest coverage added to HARD_FILTERS (min 1.5×)
SIG-12  Market cap thresholds updated to SEBI FY25 (Large≥₹30k, Mid≥₹8k)
SIG-13  None volume handled — pattern falls back to "Forming" with a clear note
SIG-15  Sideways weekly no longer counts as bullish for regime
SIG-16  Pledge-to-holding ratio check; >50% triggers early-exit penalty
SIG-17  Watchlist scan now applies min_score filter
GAP-01  FII/DII accumulation trend scored (delta from prior quarter)
GAP-02  Debt trend scored (declining D/E = quality improvement)
GAP-03  EPS consistency check (coefficient of variation of profit growth rates)
GAP-04  ROCE trend scored (rising ROCE = moat signal)
GAP-05  Piotroski F-Score (simplified, 0-9) added to fundamental scoring
GAP-06  Sector Stage2 timing context (early vs late Stage2)
GAP-07  Dividend growth scored (growing dividends = management confidence)
GAP-08  Promoter holding trend scored (buying vs selling)
GAP-10  --max-per-sector argument added to output diversification
GAP-12  Test mode includes mid/small cap compounders, not just Nifty50
"""

import os, sys, json, time, sqlite3, argparse, logging, threading
from datetime import date, datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import numpy as np

# MIN-02 FIX
warnings.filterwarnings("ignore", category=FutureWarning,      module="yfinance")
warnings.filterwarnings("ignore", category=FutureWarning,      module="pandas")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="yfinance")
warnings.filterwarnings("ignore", message=".*auto_adjust.*")
warnings.filterwarnings("ignore", message=".*Timezone.*")

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
for _d in [OUTPUT_DIR, LOG_DIR]:
    os.makedirs(_d, exist_ok=True)

_IST = timezone(timedelta(hours=5, minutes=30))
def _now():   return datetime.now(_IST)
def _today(): return _now().date()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(LOG_DIR, f"scanner_{_today()}.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("scanner")

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
MAX_WORKERS      = 4
MIN_MONTHLY_BARS = 24
MIN_MARKET_CAP_CR = 200

# Hard filter — stock fails any of these → not scored
HARD_FILTERS = {
    "min_roe":               10.0,
    "max_debt_equity":        2.0,
    "min_revenue_growth":     5.0,
    "max_pe":               120.0,
    "min_interest_coverage":  1.5,   # SIG-11 FIX: added
}

# Sector PE reference (FY25 approximate)
SECTOR_PE = {
    "Technology":             28,
    "Information Technology": 28,
    "Consumer Cyclical":      40,
    "Consumer Defensive":     50,   # SIG-03: raised from 45
    "Healthcare":             35,
    "Industrials":            30,
    "Financial Services":     18,
    "Basic Materials":        18,
    "Energy":                 15,
    "Real Estate":            30,
    "Utilities":              20,
    "Communication Services": 25,
    "default":                30,
}

# Sector EV/EBITDA reference (FY25, replaces PE-vs-sector to fix SIG-02)
SECTOR_EV_EBITDA = {
    "Technology":             20,
    "Information Technology": 20,
    "Consumer Cyclical":      25,
    "Consumer Defensive":     30,
    "Healthcare":             22,
    "Industrials":            18,
    "Financial Services":     12,
    "Basic Materials":        10,
    "Energy":                  8,
    "Real Estate":            18,
    "Utilities":              12,
    "Communication Services": 16,
    "default":                18,
}

# BUG-13 FIX: added Utilities, Communication Services, Consumer Defensive
SECTOR_INDEX = {
    "Technology":             "^CNXIT",
    "Information Technology": "^CNXIT",
    "Healthcare":             "^CNXPHARMA",
    "Consumer Cyclical":      "^CNXCONSUM",
    "Consumer Defensive":     "^CNXFMCG",    # BUG-13 FIX
    "Industrials":            "^CNXINFRA",
    "Basic Materials":        "^CNXMETAL",
    "Energy":                 "^CNXENERGY",
    "Financial Services":     "^CNXPSUBANK",
    "Real Estate":            "^CNXREALTY",
    "Utilities":              "^CNXINFRA",    # BUG-13 FIX (closest proxy)
    "Communication Services": "^CNXIT",      # BUG-13 FIX (telecom in IT index)
}

# SIG-03 FIX: sectors where higher sustainable PE is valid
HIGH_PE_SECTORS = {"Consumer Defensive", "Consumer Cyclical", "Healthcare"}

NIFTY50_SYM  = "^NSEI"
NIFTYMID_SYM = "^NSEMDCP50"
VIX_SYM      = "^INDIAVIX"

# ═══════════════════════════════════════════════════════════════════════
# DATABASE  (per-thread read connections)
# ═══════════════════════════════════════════════════════════════════════
_db_lock = threading.Lock()
_db_con  = None
_local   = threading.local()   # MIN-03 FIX


def _get_db() -> sqlite3.Connection:
    global _db_con
    with _db_lock:
        if _db_con is None:
            _db_con = sqlite3.connect(CACHE_PATH, check_same_thread=False)
            _db_con.execute("PRAGMA journal_mode=WAL")
            _db_con.execute("PRAGMA synchronous=NORMAL")
            _db_con.execute("PRAGMA cache_size=-131072")
            _db_con.executescript("""
                CREATE TABLE IF NOT EXISTS scan_results (
                    scan_id TEXT NOT NULL, scan_date TEXT NOT NULL,
                    stock TEXT NOT NULL, score REAL, grade TEXT, result_json TEXT,
                    PRIMARY KEY (scan_id, stock)
                );
                CREATE TABLE IF NOT EXISTS stock_history (
                    stock TEXT PRIMARY KEY,
                    first_seen_date TEXT, first_price REAL, first_score REAL,
                    last_seen_date TEXT,  last_price REAL,  last_score REAL,
                    times_scanned INTEGER DEFAULT 0, pattern_history TEXT
                );
            """)
            _db_con.commit()
        return _db_con


def _get_read_con() -> sqlite3.Connection:
    """MIN-03 FIX: per-thread read-only connection."""
    if not hasattr(_local, "con"):
        con = sqlite3.connect(CACHE_PATH, check_same_thread=True)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA query_only=ON")
        _local.con = con
    return _local.con


def read_cache(stock: str, tf: str = "1mo", limit: int = 9999) -> pd.DataFrame | None:
    try:
        con = _get_read_con()
        df  = pd.read_sql(
            f"SELECT date,open,high,low,close,volume FROM price_cache "
            f"WHERE stock=? AND tf=? ORDER BY date DESC LIMIT {limit}",
            con, params=(stock, tf))
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
    try:
        row = _get_read_con().execute(
            "SELECT fund_json FROM fund_cache WHERE stock=?", (stock,)
        ).fetchone()
        if row and row[0]:
            return json.loads(row[0])
    except Exception:
        pass
    return {}


def load_universe_from_cache() -> list[str]:
    try:
        rows = _get_read_con().execute(
            "SELECT DISTINCT stock FROM cache_meta WHERE tf='1mo'"
        ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS  (MIN-01 FIX: defined above scan_one)
# ═══════════════════════════════════════════════════════════════════════
def _fmt(v) -> str | float:
    """Format numeric values for CSV — blank string if None/NaN."""
    if v is None:
        return ""
    try:
        f = float(v)
        return "" if f != f else round(f, 2)
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


def _to_native(obj):
    """Convert numpy/pandas types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_to_native(v) for v in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def _ema(prices: np.ndarray, n: int) -> float:
    """
    SIG-04 FIX: Exponential Moving Average.
    Uses 3× lookback for warm-up to reduce initialisation bias.
    """
    if len(prices) == 0:
        return float("nan")
    subset = prices[-min(len(prices), n * 3):]
    k = 2.0 / (n + 1)
    ema = float(subset[0])
    for p in subset[1:]:
        ema = float(p) * k + ema * (1 - k)
    return ema


def _trend_slope(series: list) -> float | None:
    """Linear regression slope. Positive = improving."""
    clean = [float(v) for v in (series or []) if v is not None]
    if len(clean) < 2:
        return None
    x = np.arange(len(clean), dtype=float)
    try:
        return round(float(np.polyfit(x, clean, 1)[0]), 4)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# MARKET CONTEXT
# ═══════════════════════════════════════════════════════════════════════
def load_market_context() -> dict:
    ctx = {
        "regime": "Unknown", "aggression": 1, "vix": None,
        "nifty_trend_mo": "Unknown", "nifty_trend_wk": "Unknown",
        "nifty_above_10mo_ema": False, "nifty_above_30mo_ema": False,
    }

    mo = read_cache(NIFTY50_SYM, "1mo")
    if mo is not None and len(mo) >= 30:
        c = mo["Close"].values.astype(float)
        ema10 = _ema(c, 10)    # SIG-04 FIX: use EMA
        ema30 = _ema(c, 30)
        ctx["nifty_above_10mo_ema"] = bool(c[-1] > ema10)
        ctx["nifty_above_30mo_ema"] = bool(c[-1] > ema30)
        if c[-1] > ema10 > ema30:
            ctx["nifty_trend_mo"] = "Stage2-Bull"
        elif c[-1] > ema30:
            ctx["nifty_trend_mo"] = "Uptrend"
        elif c[-1] < ema10 < ema30:
            ctx["nifty_trend_mo"] = "Stage4-Bear"
        else:
            ctx["nifty_trend_mo"] = "Choppy"

    wk = read_cache(NIFTY50_SYM, "1wk")
    if wk is not None and len(wk) >= 52:
        c   = wk["Close"].values.astype(float)
        ma10 = np.mean(c[-10:]); ma40 = np.mean(c[-40:])
        if c[-1] > ma10 > ma40:
            ctx["nifty_trend_wk"] = "Uptrend"
        elif c[-1] > ma40:
            ctx["nifty_trend_wk"] = "Sideways"   # SIG-15: Sideways intentionally NOT bullish
        else:
            ctx["nifty_trend_wk"] = "Downtrend"

    vix_df = read_cache(VIX_SYM, "1mo")
    if vix_df is not None and len(vix_df) > 0:
        ctx["vix"] = round(float(vix_df["Close"].values[-1]), 2)

    mo_bull = ctx["nifty_trend_mo"] in ("Stage2-Bull", "Uptrend")
    # SIG-15 FIX: Sideways does NOT count as weekly bullish
    wk_bull = ctx["nifty_trend_wk"] == "Uptrend"
    vix_ok  = ctx["vix"] is None or ctx["vix"] < 22

    if mo_bull and wk_bull and vix_ok:
        ctx["regime"] = "Bull";     ctx["aggression"] = 3
    elif mo_bull and vix_ok:
        ctx["regime"] = "Uptrend";  ctx["aggression"] = 2
    elif mo_bull and not vix_ok:
        ctx["regime"] = "Cautious"; ctx["aggression"] = 1
    elif not mo_bull and not wk_bull:
        ctx["regime"] = "Bear";     ctx["aggression"] = 0
    else:
        ctx["regime"] = "Mixed";    ctx["aggression"] = 1

    return ctx


def get_sector_trend(sector: str) -> str:
    sym = SECTOR_INDEX.get(sector)
    if not sym:
        return "Unknown"
    df = read_cache(sym, "1mo")
    if df is None or len(df) < 12:
        return "Unknown"
    c   = df["Close"].values.astype(float)
    ema10 = _ema(c, 10)
    ema30 = _ema(c, min(30, len(c)))
    if c[-1] > ema10 > ema30:  return "Stage2"
    if c[-1] > ema30:          return "Uptrend"
    if c[-1] < ema10 < ema30:  return "Downtrend"
    return "Choppy"


def get_sector_stage_age(sector: str) -> int:
    """
    GAP-06 FIX: Count months the sector has been in Stage2.
    Returns 0 if not in Stage2. High value = late-stage (risky for new entries).
    """
    sym = SECTOR_INDEX.get(sector)
    if not sym:
        return 0
    df = read_cache(sym, "1mo")
    if df is None or len(df) < 12:
        return 0
    c   = df["Close"].values.astype(float)
    age = 0
    for i in range(len(c) - 1, max(len(c) - 48, 0), -1):
        sub = c[:i + 1]
        if len(sub) < 10:
            break
        ema10 = _ema(sub, 10)
        ema30 = _ema(sub, min(30, len(sub)))
        if sub[-1] > ema10 > ema30:
            age += 1
        else:
            break
    return age


# ═══════════════════════════════════════════════════════════════════════
# VOLUME SURGE HELPER
# ═══════════════════════════════════════════════════════════════════════
def _vsurge_mo(v: np.ndarray | None, lb: int = 12) -> float | None:
    """
    BUG-01 FIX: exclude current bar (v[-1]) from the baseline average.
    Use v[-lb-1:-1] so the current month does not contaminate its own baseline.
    """
    if v is None or len(v) < lb + 1:
        return None
    avg = np.mean(v[-lb - 1:-1])   # BUG-01 FIX
    return round(float(v[-1] / avg), 2) if avg > 0 else None


# ═══════════════════════════════════════════════════════════════════════
# MONTHLY PATTERN DETECTORS
# All accept: c (close array), v (volume array),
#             h (high array, optional), lo (low array, optional)
# Return dict or None.
# ═══════════════════════════════════════════════════════════════════════

def det_monthly_cup(c: np.ndarray, v: np.ndarray | None,
                    h: np.ndarray | None = None,
                    lo: np.ndarray | None = None) -> dict | None:
    """
    Multi-year Cup & Handle on monthly chart.
    SIG-05 FIX: uses high array for peak detection.
    SIG-10 FIX: verifies prior uptrend ≥ 20% before the cup.
    """
    n = len(c)
    if n < 24:
        return None
    # SIG-05 FIX: use highs for smoothed peak detection if available
    arr = h if h is not None else c
    s   = pd.Series(arr).rolling(3, min_periods=1).mean().values

    best = None
    for cup_len in range(12, min(49, n - 2)):
        seg    = s[n - cup_len:]
        c_seg  = c[n - cup_len:]
        ti     = int(np.argmin(seg))
        if not (int(cup_len * 0.25) <= ti <= int(cup_len * 0.75)):
            continue

        lm  = np.max(seg[:ti + 1])
        rm  = np.max(seg[ti:])
        pk  = max(lm, rm)
        tr  = seg[ti]

        depth = (pk - tr) / pk if pk > 0 else 1
        if not (0.18 <= depth <= 0.65):
            continue
        sym = abs(lm - rm) / pk if pk > 0 else 1
        if sym > 0.25:
            continue

        # Handle: last 2–8 months, mild retracement
        handle_len = max(2, min(6, cup_len // 4))
        handle     = seg[-handle_len:]
        handle_drop = ((np.max(handle) - np.min(handle)) / np.max(handle)
                       if np.max(handle) > 0 else 1)
        if handle_drop > 0.30:
            continue

        # Parabolic cup shape via quadratic fit
        try:
            x   = np.arange(ti + 1, dtype=float)
            cf  = np.polyfit(x, seg[:ti + 1], 2)
            ssr = np.sum((seg[:ti + 1] - np.polyval(cf, x)) ** 2)
            sst = np.sum((seg[:ti + 1] - np.mean(seg[:ti + 1])) ** 2)
            r2  = 1 - ssr / sst if sst > 0 else 0
            if cf[0] <= 0 or r2 < 0.45:
                continue
        except Exception:
            continue

        # SIG-10 FIX: verify prior uptrend >= 20% before the cup
        prior_start = n - cup_len - 12
        if prior_start >= 0:
            prior_gain = (c[n - cup_len] - c[prior_start]) / max(c[prior_start], 1)
        else:
            prior_gain = (c[n - cup_len] - c[0]) / max(c[0], 1)
        if prior_gain < 0.20:
            continue   # SIG-10: not a valid cup without prior uptrend

        q = r2 * (1 - sym) * (1 - handle_drop * 0.5)
        if best is None or q > best["q"]:
            best = dict(q=q, pk=pk, tr=tr, depth=depth, sym=sym,
                        cup_len=cup_len, handle_len=handle_len, r2=r2,
                        prior_gain=prior_gain)

    if best is None:
        return None

    vs  = _vsurge_mo(v)
    bz  = round(float(best["pk"]), 2)
    # SIG-13 FIX: if no volume data, pattern is "Forming" not "Breakout Ready"
    bo  = (c[-1] >= bz * 0.97) and (vs is not None and vs >= 1.3)
    return dict(
        pattern="MonthlyCupHandle",
        status="Breakout Ready" if bo else "Forming",
        quality=round(best["q"], 3),
        bz=bz, bottom=round(float(best["tr"]), 2),
        last=round(float(c[-1]), 2),
        duration_months=best["cup_len"],
        details=(f"Cup {best['cup_len']}mo | Depth {best['depth']*100:.0f}% | "
                 f"Prior gain {best['prior_gain']*100:.0f}% | R²={best['r2']:.2f}"),
        vs=vs,
    )


def det_monthly_vcp(c: np.ndarray, v: np.ndarray | None,
                    h: np.ndarray | None = None,
                    lo: np.ndarray | None = None) -> dict | None:
    """
    Volatility Contraction Pattern — monthly chart.
    BUG-04 FIX: quality capped at 1.0.
    BUG-05 FIX: contraction threshold = 0.80 (was 0.85).
    SIG-05 FIX: uses High for peaks, Low for troughs.
    """
    n = len(c)
    if n < 18:
        return None

    # SIG-05 FIX: use High/Low if available
    h_arr  = h  if h  is not None else c
    lo_arr = lo if lo is not None else c
    prom   = max(np.mean(np.abs(np.diff(c))) * 1.5, np.mean(c) * 0.015)

    try:
        highs, _ = find_peaks(h_arr,  prominence=prom, distance=3)
        lows,  _ = find_peaks(-lo_arr, prominence=prom, distance=3)
    except Exception:
        return None

    if len(highs) < 2 or len(lows) < 2:
        return None

    contractions = []
    hl = list(highs) + [n]
    for i, hi in enumerate(hl[:-1]):
        nh = hl[i + 1]
        nl = lows[(lows > hi) & (lows < nh)]
        lo_i = (nl[0] if len(nl) > 0
                else hi + int(np.argmin(lo_arr[hi:nh])) if nh - hi >= 3 else -1)
        if lo_i < 0 or lo_i >= n:
            continue
        depth = (h_arr[hi] - lo_arr[lo_i]) / h_arr[hi] if h_arr[hi] > 0 else 0
        if depth < 0.04:
            continue
        contractions.append((hi, lo_i, depth))

    if len(contractions) < 3:
        return None

    depths = [ct[2] for ct in contractions]
    # BUG-05 FIX: 0.80 (was 0.85) — each contraction ≤80% of prior
    if not all(depths[i] <= depths[i - 1] * 0.80 for i in range(1, len(depths))):
        return None

    if contractions[-1][1] < int(n * 0.5):
        return None

    final_depth = depths[-1]
    if final_depth > 0.12:
        return None

    pivot = float(np.max(h_arr[highs]))
    vs    = _vsurge_mo(v)
    bo    = (c[-1] >= pivot * 0.97) and (vs is not None and vs >= 1.4)

    # BUG-04 FIX: cap quality at 1.0
    quality = round(min(1.0, (1 - final_depth) * (len(contractions) / 5)), 3)

    return dict(
        pattern="MonthlyVCP",
        status="Breakout Ready" if bo else "Forming",
        quality=quality,
        bz=round(pivot, 2),
        bottom=round(float(lo_arr[contractions[-1][1]]), 2),
        last=round(float(c[-1]), 2),
        duration_months=contractions[-1][1] - contractions[0][0],
        details=(f"{len(contractions)} contractions | "
                 f"First {depths[0]*100:.0f}% → Final {final_depth*100:.0f}%"),
        vs=vs,
    )


def det_monthly_flat_base(c: np.ndarray, v: np.ndarray | None,
                           h: np.ndarray | None = None,
                           lo: np.ndarray | None = None) -> dict | None:
    """
    Multi-year Flat Base on monthly chart.
    BUG-06 FIX: uses continue not break — checks ALL lengths, not just short ones.
    """
    n   = len(c)
    if n < 12:
        return None
    h_arr  = h  if h  is not None else c
    lo_arr = lo if lo is not None else c

    best = None
    for bl in range(6, min(25, n) + 1):
        bh  = np.max(h_arr[-bl:])
        blo = np.min(lo_arr[-bl:])
        rng = (bh - blo) / bh if bh > 0 else 1

        # BUG-06 FIX: continue (not break) so all lengths are checked
        if rng > 0.15:
            continue

        # Prior uptrend must be ≥15%
        if n - bl >= 12:
            prior_gain = (c[n - bl] - c[n - bl - 12]) / max(c[n - bl - 12], 1)
        elif n - bl >= 1:
            prior_gain = (c[n - bl] - c[0]) / max(c[0], 1)
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
    bo = (c[-1] >= bz * 0.98) and (vs is not None and vs >= 1.2)

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


def det_monthly_stage2(c: np.ndarray, v: np.ndarray | None,
                        h: np.ndarray | None = None,
                        lo: np.ndarray | None = None) -> dict | None:
    """
    Weinstein Stage 2 on monthly chart.
    SIG-04 FIX: uses 10-month EMA (not SMA) — institutional standard.
    BUG-07 FIX: quality normalised to 0–1 scale (was capped at 0.36).
    """
    n = len(c)
    if n < 30:
        return None

    # SIG-04 FIX: 10-month EMA for current, and lagged for trend direction
    ema10_now  = _ema(c, 10)
    ema30_now  = _ema(c, 30)
    ema10_prev = _ema(c[:-3], 10) if n >= 13 else ema10_now
    ema30_prev = _ema(c[:-12], 30) if n >= 42 else ema30_now

    if c[-1] <= ema30_now:
        return None

    ma_turning = ema30_now >= ema30_prev * 0.97

    # Must have been below 30-EMA within last 12 months
    recently_below = any(
        c[i] < _ema(c[:i + 1], min(30, i + 1))
        for i in range(max(0, n - 12), n - 1)
    )

    if not recently_below and not (ema30_now < ema30_prev * 0.98):
        return None

    vs = _vsurge_mo(v)
    bo = vs is not None and vs >= 1.5

    # BUG-07 FIX: normalise quality to 0–1 (was min(dist/MA, 0.30) × 1.2 = max 0.36)
    dist_ratio = min((c[-1] - ema30_now) / max(ema30_now * 0.15, 1), 1.0)
    quality    = round(dist_ratio * (1.0 if recently_below else 0.7), 3)

    return dict(
        pattern="MonthlyStage2",
        status="Breaking Out" if bo else "Stage Change",
        quality=quality,
        bz=round(float(ema30_now), 2),
        bottom=round(float(np.min(c[-24:])), 2),
        last=round(float(c[-1]), 2),
        duration_months=24,
        details=(f"10-EMA={ema10_now:.0f} | 30-EMA={ema30_now:.0f} | "
                 f"Above by {(c[-1]-ema30_now)/ema30_now*100:.1f}% | "
                 f"MA turning={'Yes' if ma_turning else 'No'}"),
        vs=vs,
    )


def det_monthly_asc_triangle(c: np.ndarray, v: np.ndarray | None,
                              h: np.ndarray | None = None,
                              lo: np.ndarray | None = None) -> dict | None:
    """
    Ascending Triangle — monthly chart.
    BUG-08 FIX: resistance tolerance raised to 10% (was 5%).
    SIG-05 FIX: uses High for peaks, Low for troughs.
    """
    n = len(c)
    if n < 18:
        return None

    h_arr  = h  if h  is not None else c
    lo_arr = lo if lo is not None else c
    prom   = max(np.mean(np.abs(np.diff(c))) * 1.2, np.mean(c) * 0.01)

    try:
        pks, _ = find_peaks(h_arr,   prominence=prom, distance=3)
        trs, _ = find_peaks(-lo_arr, prominence=prom, distance=3)
    except Exception:
        return None

    if len(pks) < 2 or len(trs) < 2:
        return None

    pp  = h_arr[pks]
    res = np.median(pp)
    # BUG-08 FIX: 10% tolerance (was 5%) — realistic for monthly charts
    if (np.max(pp) - np.min(pp)) / max(res, 1) > 0.10:
        return None

    tp     = lo_arr[trs]
    slopes = [(tp[j] - tp[i]) / max(trs[j] - trs[i], 1)
              for i in range(len(trs)) for j in range(i + 1, len(trs))]
    if not slopes or np.median(slopes) <= 0:
        return None

    rise = (tp[-1] - tp[0]) / max(tp[0], 1) if tp[0] > 0 else 0
    if rise < 0.10 or trs[-1] < int(n * 0.4):
        return None

    vs = _vsurge_mo(v)
    bz = round(float(res), 2)
    bo = (c[-1] >= bz * 0.98) and (vs is not None and vs >= 1.2)

    return dict(
        pattern="MonthlyAscTriangle",
        status="Breakout Ready" if bo else "Forming",
        quality=round(min(1.0, rise * (1 - (np.max(pp) - np.min(pp)) / max(res, 1))), 3),
        bz=bz,
        bottom=round(float(tp[0]), 2),
        last=round(float(c[-1]), 2),
        duration_months=int(trs[-1] - pks[0]),
        details=(f"{len(pks)} resistance tests | Support rise {rise*100:.0f}% | "
                 f"Resistance flat {(np.max(pp)-np.min(pp))/max(res,1)*100:.1f}%"),
        vs=vs,
    )


def det_monthly_double_bottom(c: np.ndarray, v: np.ndarray | None,
                               h: np.ndarray | None = None,
                               lo: np.ndarray | None = None) -> dict | None:
    """Multi-year W-bottom on monthly chart. SIG-05 FIX: uses Low for troughs."""
    n = len(c)
    if n < 24:
        return None

    lo_arr = lo if lo is not None else c
    try:
        troughs, _ = find_peaks(-lo_arr, prominence=0.05 * np.mean(c), distance=4)
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
            p1, p2 = lo_arr[troughs[i]], lo_arr[troughs[j]]
            diff   = abs(p1 - p2) / max(min(p1, p2), 1)
            if diff > 0.10:
                continue
            h_seg  = h if h is not None else c
            mid_max = np.max(h_seg[troughs[i]:troughs[j] + 1])
            mr      = (mid_max - (p1 + p2) / 2) / max((p1 + p2) / 2, 1)
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
    bo = (c[-1] >= bz * 0.98) and (vs is not None and vs >= 1.2)

    return dict(
        pattern="MonthlyDoubleBottom",
        status="Breakout Ready" if bo else "Forming",
        quality=round(min(1.0, best["sc"]), 3),
        bz=bz,
        bottom=round(float(best["bottom"]), 2),
        last=round(float(c[-1]), 2),
        duration_months=best["sep"],
        details=(f"Sep {best['sep']}mo | Mid rally {best['mr']*100:.0f}% | "
                 f"Asymmetry {best['diff']*100:.0f}%"),
        vs=vs,
    )


# ═══════════════════════════════════════════════════════════════════════
# WEEKLY & DAILY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
def analyze_weekly(wk_df: pd.DataFrame | None) -> dict:
    r = {"trend": "Unknown", "above_40wma": False, "vol_expanding": False, "score": 0}
    if wk_df is None or len(wk_df) < 20:
        return r
    c = wk_df["Close"].values.astype(float)
    v = wk_df["Volume"].values.astype(float)
    n = len(c)
    ma10 = np.mean(c[-10:]) if n >= 10 else c[-1]
    ma40 = np.mean(c[-40:]) if n >= 40 else np.mean(c)
    r["above_40wma"] = bool(c[-1] > ma40)
    if c[-1] > ma10 > ma40:
        r["trend"] = "Uptrend";           r["score"] = 8
    elif c[-1] > ma40:
        r["trend"] = "Sideways-Above-MA"; r["score"] = 5
    elif c[-1] > ma10:
        r["trend"] = "Sideways";          r["score"] = 3
    else:
        r["trend"] = "Downtrend";         r["score"] = 0
    if n >= 20:
        vol_up = np.mean(v[-4:]) if len(v) >= 4 else 0
        vol_all = np.mean(v[-20:]) if len(v) >= 20 else 0
        up_recent = sum(1 for i in range(-4, 0) if c[i] > c[i - 1])
        r["vol_expanding"] = bool(vol_up > vol_all * 1.1 and up_recent >= 3)
    return r


def analyze_daily(d_df: pd.DataFrame | None) -> dict:
    r = {"setup": "Unknown", "above_50dma": False, "tightening": False, "score": 0}
    if d_df is None or len(d_df) < 30:
        return r
    c = d_df["Close"].values.astype(float)
    n = len(c)
    ma50  = np.mean(c[-50:])  if n >= 50  else np.mean(c)
    ma200 = np.mean(c[-200:]) if n >= 200 else np.mean(c)
    r["above_50dma"] = bool(c[-1] > ma50)
    if n >= 30:
        rr  = (np.max(c[-10:]) - np.min(c[-10:])) / max(np.mean(c[-10:]), 1)
        pr  = (np.max(c[-30:-10]) - np.min(c[-30:-10])) / max(np.mean(c[-30:-10]), 1)
        r["tightening"] = bool(rr < pr * 0.7)
    if c[-1] > ma50 > ma200:
        r["setup"] = "Tight-Above-MAs" if r["tightening"] else "Uptrend"
        r["score"] = 4 if r["tightening"] else 2
    elif c[-1] > ma200:
        r["setup"] = "Constructive"; r["score"] = 1
    else:
        r["setup"] = "Below-MAs";    r["score"] = 0
    return r


# ═══════════════════════════════════════════════════════════════════════
# RELATIVE STRENGTH
# ═══════════════════════════════════════════════════════════════════════
def calc_rs_vs_nifty(stock_mo: pd.DataFrame, n50_mo: pd.DataFrame,
                     years: int = 1) -> float | None:
    months = years * 12
    try:
        combined = pd.DataFrame({"s": stock_mo["Close"], "n": n50_mo["Close"]}).dropna()
        if len(combined) < months:
            return None
        sr = (combined["s"].iloc[-1] / combined["s"].iloc[-months] - 1) * 100
        nr = (combined["n"].iloc[-1] / combined["n"].iloc[-months] - 1) * 100
        return round(float(sr - nr), 2)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# PIOTROSKI F-SCORE  (GAP-05 FIX)
# ═══════════════════════════════════════════════════════════════════════
def compute_piotroski(fund: dict) -> tuple[int, list[str]]:
    """
    Simplified Piotroski F-Score (0-9) using available data.
    Profitability (3): ROA>0, OCF>0, cash quality (OCF>net income)
    Leverage (3):      D/E declining, current ratio >1.5, interest coverage >2
    Efficiency (3):    ROCE rising, OPM rising, revenue growing
    """
    score = 0
    notes: list[str] = []

    # ── Profitability ─────────────────────────────────────────────────────────
    roa = fund.get("roa")
    if roa and roa > 0:
        score += 1

    ocf = fund.get("operating_cashflow")
    if ocf and ocf > 0:
        score += 1

    fcf = fund.get("free_cashflow")
    if ocf and ocf > 0 and fcf and fcf > 0 and ocf >= fcf * 0.8:
        score += 1   # Cash quality: OCF substantive relative to FCF

    # ── Leverage ──────────────────────────────────────────────────────────────
    de_trend = fund.get("de_trend")
    if de_trend is not None and de_trend < 0:
        score += 1   # D/E declining = improving leverage

    cr = fund.get("current_ratio") or fund.get("current_ratio_screener")
    if cr and cr > 1.5:
        score += 1

    icr = fund.get("interest_coverage")
    if icr and icr > 2.0:
        score += 1

    # ── Operating Efficiency ──────────────────────────────────────────────────
    roce_trend = fund.get("roce_trend")
    if roce_trend is not None and roce_trend > 0:
        score += 1   # ROCE rising = moat expanding

    opm_trend = fund.get("opm_trend")
    if opm_trend is not None and opm_trend > 0:
        score += 1   # Margin expanding

    rev = fund.get("sales_3yr_cagr") or fund.get("revenue_growth_ttm")
    if rev and rev > 10:
        score += 1   # Revenue growing meaningfully

    if score >= 7:
        notes.append(f"Piotroski F={score} (high quality compounder)")
    elif score >= 5:
        notes.append(f"Piotroski F={score} (above average quality)")

    return score, notes


# ═══════════════════════════════════════════════════════════════════════
# EPS CONSISTENCY  (GAP-03 FIX)
# ═══════════════════════════════════════════════════════════════════════
def compute_eps_consistency(fund: dict) -> tuple[float, str]:
    """
    Coefficient of variation of annual profit growth rates.
    Low CV = steady compounder; High CV = lumpy / cyclical earner.
    Returns (score 0-1, description).
    """
    profit_history = fund.get("profit_history", [])
    if not profit_history or len(profit_history) < 3:
        return 0.5, "insufficient history"

    vals = [float(v) if v is not None else None for v in profit_history]

    # Check for loss years
    loss_years = sum(1 for v in vals if v is not None and v < 0)
    if loss_years >= 2:
        return 0.2, "multiple loss years in history"

    pos = [v for v in vals if v and v > 0]
    if len(pos) < 2:
        return 0.3, "insufficient profitable years"

    growth_rates = [(pos[i] - pos[i - 1]) / abs(pos[i - 1])
                    for i in range(1, len(pos)) if pos[i - 1] != 0]
    if len(growth_rates) < 2:
        return 0.5, "insufficient data"

    mean_g = np.mean(growth_rates)
    std_g  = np.std(growth_rates)
    cv     = std_g / abs(mean_g) if abs(mean_g) > 0.01 else 1.0

    if cv < 0.30 and mean_g > 0.15:
        return 1.0, "steady compounder (low CV)"
    elif cv < 0.50 and mean_g > 0.10:
        return 0.75, "consistent growth"
    elif cv < 0.80:
        return 0.40, "lumpy earner"
    else:
        return 0.15, "highly inconsistent earnings"


# ═══════════════════════════════════════════════════════════════════════
# SCORING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def score_fundamentals(fund: dict) -> tuple[float, list[str]]:
    """
    Score fundamentals 0–40 pts (capped).
    All audit fixes applied:
    SIG-01: penalty for negative/very-low profit growth
    SIG-06: FCF scored as yield tiers (0-6 pts)
    SIG-07: OPM trend penalty
    SIG-16: pledge-to-holding ratio check with early exit
    GAP-01: FII/DII accumulation trend
    GAP-02: debt trend bonus
    GAP-03: EPS consistency
    GAP-04: ROCE trend bonus
    GAP-05: Piotroski F-Score bonus
    GAP-07: dividend growth bonus
    GAP-08: promoter accumulation trend
    """
    score:   float     = 0.0
    reasons: list[str] = []

    # ── SIG-16 FIX: Early-exit for extreme pledge risk ───────────────────────
    ph     = fund.get("promoter_holding") or 0
    pledge = fund.get("promoter_pledge",  0) or 0
    if ph > 0:
        pledge_ratio = pledge / ph
        if pledge_ratio > 0.50:
            score -= 6
            reasons.append(f"⚠️ {pledge:.0f}% pledged ({pledge_ratio*100:.0f}% of holding)")
            return round(max(0, min(score, 40)), 2), reasons

    # ── Revenue Growth (0-8 pts) ──────────────────────────────────────────────
    rev = fund.get("sales_3yr_cagr") or fund.get("revenue_growth_ttm")
    if rev is not None:
        if rev > 25:   score += 8; reasons.append(f"Revenue CAGR {rev:.0f}%")
        elif rev > 20: score += 6; reasons.append(f"Revenue CAGR {rev:.0f}%")
        elif rev > 15: score += 4; reasons.append(f"Revenue CAGR {rev:.0f}%")
        elif rev > 10: score += 2
        elif rev > 5:  score += 1

    # ── Profit Growth (0-8 pts) + SIG-01 penalty ─────────────────────────────
    profit_growth = fund.get("profit_3yr_cagr") or fund.get("earnings_growth_ttm")
    if profit_growth is not None:
        if profit_growth > 30:   score += 8; reasons.append(f"Profit CAGR {profit_growth:.0f}%")
        elif profit_growth > 25: score += 6; reasons.append(f"Profit CAGR {profit_growth:.0f}%")
        elif profit_growth > 20: score += 4; reasons.append(f"Profit CAGR {profit_growth:.0f}%")
        elif profit_growth > 15: score += 2
        # SIG-01 FIX: penalty for declining / very-low profit
        elif profit_growth < 0:
            score -= 4
            reasons.append(f"Declining profits ({profit_growth:.0f}% CAGR) ⚠️")
        elif profit_growth < 5:
            score -= 2

    # ── ROE (0-6 pts) ─────────────────────────────────────────────────────────
    roe = fund.get("roe_screener") or fund.get("roe")
    if roe is not None:
        if roe > 25:   score += 6; reasons.append(f"ROE {roe:.0f}%")
        elif roe > 20: score += 4; reasons.append(f"ROE {roe:.0f}%")
        elif roe > 15: score += 2
        elif roe > 12: score += 1

    # ── ROCE (0-6 pts) + GAP-04 trend bonus ──────────────────────────────────
    roce = fund.get("roce")
    if roce is not None:
        if roce > 25:   score += 6; reasons.append(f"ROCE {roce:.0f}%")
        elif roce > 20: score += 4; reasons.append(f"ROCE {roce:.0f}%")
        elif roce > 15: score += 2
        elif roce > 12: score += 1
    # GAP-04 FIX: ROCE trend bonus (rising ROCE = expanding moat)
    roce_trend = fund.get("roce_trend")
    if roce_trend is not None and roce_trend > 0:
        score += 2; reasons.append("ROCE trend rising ↑")
    elif roce_trend is not None and roce_trend < -1:
        score -= 1

    # ── Debt Management (0-6 pts) + GAP-02 trend bonus ───────────────────────
    de = fund.get("debt_to_equity_ratio")
    if de is not None:
        if de < 0.10:  score += 6; reasons.append("Nearly debt-free")
        elif de < 0.30: score += 5; reasons.append(f"Low D/E {de:.2f}")
        elif de < 0.50: score += 4
        elif de < 1.00: score += 2
        elif de < 1.50: score += 0
        else:           score -= 2
    # GAP-02 FIX: Debt trend (de_trend negative = D/E falling = improving)
    de_trend = fund.get("de_trend")
    if de_trend is not None and de_trend < 0 and (de or 999) < 1.5:
        score += 2; reasons.append("Debt declining ↓")
    elif de_trend is not None and de_trend > 0.05:
        score -= 1   # Rising debt = caution

    # ── Promoter Holding (0-4 pts) + GAP-08 trend ────────────────────────────
    # (pledge already handled above in SIG-16 early-exit)
    if ph > 0:
        if ph > 60 and pledge < 5:    score += 4; reasons.append(f"Promoter {ph:.0f}% holding")
        elif ph > 50 and pledge < 10: score += 3; reasons.append(f"Promoter {ph:.0f}% holding")
        elif ph > 40 and pledge < 20: score += 2
        elif pledge > 30:             score -= 2

    # GAP-08 FIX: Promoter accumulation trend
    pro_delta = fund.get("promoter_delta")
    if pro_delta is not None:
        if pro_delta > 1.0:   score += 2; reasons.append("Promoters buying ↑")
        elif pro_delta < -2.0: score -= 1

    # ── Operating Margin (0-4 pts) + SIG-07 trend penalty ────────────────────
    opm = fund.get("operating_margin_screener") or fund.get("operating_margin")
    if opm is not None:
        if opm > 25:   score += 4; reasons.append(f"OPM {opm:.0f}%")
        elif opm > 20: score += 3
        elif opm > 15: score += 2
        elif opm > 10: score += 1
    # SIG-07 FIX: OPM trend penalty
    opm_trend = fund.get("opm_trend")
    if opm_trend is not None and opm_trend < -0.5:
        score -= 2; reasons.append("Margin compressing ↓ ⚠️")

    # ── FCF Yield (0-6 pts) — SIG-06 FIX: tiered yield, not binary ───────────
    fcf = fund.get("free_cashflow")
    mc  = fund.get("market_cap") or 0
    if fcf is not None and mc > 0:
        fcf_yield = (fcf / mc) * 100
        if fcf_yield > 5:    score += 6; reasons.append(f"FCF yield {fcf_yield:.1f}%")
        elif fcf_yield > 3:  score += 4; reasons.append(f"FCF yield {fcf_yield:.1f}%")
        elif fcf_yield > 1:  score += 2
        elif fcf > 0:        score += 1
        else:                score -= 2  # Negative FCF is a warning
    elif fcf and fcf > 0:
        score += 1   # Positive FCF but can't compute yield

    # ── EPS Consistency (0-2 pts) — GAP-03 FIX ──────────────────────────────
    eps_con, eps_desc = compute_eps_consistency(fund)
    if eps_con >= 0.75:
        score += 2; reasons.append(f"EPS: {eps_desc}")
    elif eps_con >= 0.40:
        score += 1
    elif eps_con < 0.25:
        score -= 1

    # ── Piotroski F-Score bonus (0-3 pts) — GAP-05 FIX ──────────────────────
    f_score, f_notes = compute_piotroski(fund)
    if f_score >= 7:   score += 3; reasons += f_notes
    elif f_score >= 5: score += 1; reasons += f_notes
    elif f_score <= 3: score -= 1

    # ── FII / DII Accumulation (0-2 pts) — GAP-01 FIX ───────────────────────
    fii_d = fund.get("fii_delta")
    dii_d = fund.get("dii_delta")
    if fii_d is not None and fii_d > 0.5:
        score += 1; reasons.append(f"FII buying +{fii_d:.1f}% ↑")
    if dii_d is not None and dii_d > 0.5:
        score += 1; reasons.append(f"DII buying +{dii_d:.1f}% ↑")
    if fii_d is not None and fii_d < -1.0:
        score -= 1

    # ── Dividend Growth (0-1 pt) — GAP-07 FIX ────────────────────────────────
    div_g = fund.get("dividend_growth_3yr")
    if div_g is not None and div_g > 10:
        score += 1; reasons.append(f"Dividend growing {div_g:.0f}%/yr")

    return round(max(0.0, min(score, 40.0)), 2), reasons


def score_technical(pattern: dict | None, wk: dict, d: dict,
                    market_ctx: dict, sector_trend: str,
                    sector_stage_age: int = 0) -> tuple[float, list[str]]:
    """
    Score technical setup 0–30 pts.
    BUG-09 FIX: "Strong-Bull" dead code removed.
    GAP-06 FIX: late-stage sector penalty.
    SIG-13 FIX: None volume handled gracefully.
    """
    score:   float     = 0.0
    reasons: list[str] = []

    # ── Monthly Pattern (0-15 pts) ────────────────────────────────────────────
    if pattern:
        q      = pattern.get("quality", 0)
        status = pattern.get("status", "")
        pat    = pattern.get("pattern", "")
        dur    = pattern.get("duration_months", 0)
        vs     = pattern.get("vs")

        base = round(min(q * 12, 12), 2)

        if "Breaking Out" in status or "Breakout Ready" in status:
            base = min(15, base + 3)
            reasons.append(f"{pat} — {status}")
        else:
            reasons.append(f"{pat} forming ({dur}mo base)")

        if dur >= 24:   base = min(15, base + 1.5)
        elif dur >= 18: base = min(15, base + 1.0)
        elif dur >= 12: base = min(15, base + 0.5)

        # SIG-13 FIX: note when volume data is absent
        if vs is None:
            reasons.append("Note: no volume data — breakout confirmation unavailable")

        score += base

    # ── Weekly Confirmation (0-8 pts) ─────────────────────────────────────────
    score += wk.get("score", 0)
    if wk["trend"] == "Uptrend":   reasons.append("Weekly uptrend")
    if wk.get("vol_expanding"):    reasons.append("Weekly volume expanding")

    # ── Daily Setup (0-4 pts) ─────────────────────────────────────────────────
    score += d.get("score", 0)
    if d["setup"] == "Tight-Above-MAs":
        reasons.append("Tight daily consolidation above MAs")

    # ── Index / Sector Alignment (0-3 pts) ────────────────────────────────────
    # BUG-09 FIX: only "Bull" (no "Strong-Bull" dead code)
    regime = market_ctx.get("regime", "Unknown")
    if regime == "Bull":
        score += 3; reasons.append("Bull market")
    elif regime in ("Uptrend", "Cautious"):
        score += 2
    elif regime == "Bear":
        score = max(0, score - 2)

    if sector_trend == "Stage2":
        score = min(30, score + 1)
        reasons.append("Sector in Stage2")

    # GAP-06 FIX: late Stage2 penalty (sector extended for >24 months)
    if sector_stage_age > 24:
        score = max(0, score - 1)
        reasons.append(f"Sector Stage2 aged {sector_stage_age}mo — late-stage caution")

    return round(min(score, 30), 2), reasons


def score_valuation(fund: dict, cmp: float) -> tuple[float, list[str]]:
    """
    Score valuation 0–20 pts.
    SIG-02 FIX: PE vs Sector replaced with EV/EBITDA vs sector.
    SIG-03 FIX: sustainable PE raised for FMCG / Consumer names.
    SIG-08 FIX: analyst target gated on ≥3 analysts.
    """
    score:   float     = 0.0
    reasons: list[str] = []

    sector  = fund.get("sector", "")
    pe      = fund.get("pe_ratio") or fund.get("pe_screener")
    growth  = fund.get("profit_3yr_cagr") or fund.get("earnings_growth_ttm")
    roe     = fund.get("roe_screener") or fund.get("roe") or 15

    # ── PEG Ratio (0-8 pts) ───────────────────────────────────────────────────
    if pe and pe > 0 and growth and growth > 0:
        peg = pe / growth
        fund["_peg"] = round(peg, 2)
        if peg < 0.5:   score += 8; reasons.append(f"PEG {peg:.2f} — deeply undervalued")
        elif peg < 1.0: score += 6; reasons.append(f"PEG {peg:.2f} — undervalued")
        elif peg < 1.5: score += 4; reasons.append(f"PEG {peg:.2f} — fair value")
        elif peg < 2.0: score += 2

    # ── EV/EBITDA vs Sector (0-4 pts) — SIG-02 FIX: replaces PE vs sector ────
    ev_ebitda    = fund.get("ev_ebitda")
    sector_ev    = SECTOR_EV_EBITDA.get(sector, SECTOR_EV_EBITDA["default"])
    if ev_ebitda and ev_ebitda > 0:
        ev_ratio = ev_ebitda / sector_ev
        if ev_ratio < 0.70:   score += 4; reasons.append(f"EV/EBITDA {ev_ebitda:.0f}× below sector ({sector_ev}×)")
        elif ev_ratio < 0.90: score += 3; reasons.append(f"EV/EBITDA {ev_ebitda:.0f}× vs sector {sector_ev}×")
        elif ev_ratio < 1.10: score += 2
        elif ev_ratio > 1.50: score -= 1   # Premium to sector

    # ── PB vs Justified (0-4 pts) ─────────────────────────────────────────────
    pb = fund.get("pb_ratio")
    if pb and pb > 0:
        justified_pb = max(1.0, roe / 15)
        if pb < justified_pb * 0.7:   score += 4; reasons.append(f"PB {pb:.1f} below justified {justified_pb:.1f}")
        elif pb < justified_pb:       score += 2
        elif pb < justified_pb * 1.5: score += 1

    # ── Analyst Upside (0-4 pts) — SIG-08 FIX: minimum 3 analysts ───────────
    tgt   = fund.get("analyst_target")
    count = fund.get("analyst_count") or 0
    if tgt and cmp > 0 and count >= 3:   # SIG-08 FIX
        upside = (tgt - cmp) / cmp * 100
        if upside > 40:   score += 4; reasons.append(f"Analyst target ₹{tgt:.0f} (+{upside:.0f}%, n={count})")
        elif upside > 25: score += 3; reasons.append(f"Analyst target ₹{tgt:.0f} (+{upside:.0f}%)")
        elif upside > 15: score += 2
        elif upside > 0:  score += 1

    return round(min(score, 20), 2), reasons


def score_momentum(mo_df: pd.DataFrame | None,
                   n50_mo: pd.DataFrame | None,
                   fund: dict) -> tuple[float, list[str]]:
    """Score momentum 0–10 pts."""
    score:   float     = 0.0
    reasons: list[str] = []

    rs_1yr = None
    if mo_df is not None and n50_mo is not None:
        rs_1yr = calc_rs_vs_nifty(mo_df, n50_mo, years=1)
    if rs_1yr is not None:
        if rs_1yr > 30:   score += 6; reasons.append(f"Outperforms Nifty by {rs_1yr:.0f}%/yr")
        elif rs_1yr > 20: score += 4; reasons.append(f"Outperforms Nifty by {rs_1yr:.0f}%/yr")
        elif rs_1yr > 10: score += 2
        elif rs_1yr > 0:  score += 1

    wkh = fund.get("fifty_two_week_high")
    cmp = fund.get("regular_market_price")
    if wkh and cmp and wkh > 0:
        dist = (wkh - cmp) / wkh * 100
        if dist < 5:    score += 4; reasons.append("Near 52-week high")
        elif dist < 15: score += 2
        elif dist < 25: score += 1

    return round(min(score, 10), 2), reasons


# ═══════════════════════════════════════════════════════════════════════
# LONG-TERM TARGET CALCULATION
# ═══════════════════════════════════════════════════════════════════════
def calc_lt_targets(cmp: float, fund: dict, pattern: dict | None) -> dict:
    """
    BUG-02 FIX: 3yr exponent is 3.0 (was 2.5).
    BUG-03 FIX: 1yr target uses forward EPS × sustainable PE.
    SIG-03 FIX: sustainable PE raised for FMCG/Consumer names.
    """
    result = {
        "target_1yr": None, "target_2yr": None, "target_3yr": None,
        "fair_value":  None, "stop_loss": None,
        "expected_cagr_2yr": None, "upside_potential_pct": None,
    }
    if cmp <= 0:
        return result

    # Stop loss: 15% below entry OR above pattern bottom
    if pattern and pattern.get("bottom") and pattern["bottom"] > 0:
        result["stop_loss"] = round(max(cmp * 0.85, pattern["bottom"] * 0.95), 2)
    else:
        result["stop_loss"] = round(cmp * 0.85, 2)

    pe       = fund.get("pe_ratio") or fund.get("pe_screener")
    growth   = fund.get("profit_3yr_cagr") or fund.get("earnings_growth_ttm") or 15
    sector   = fund.get("sector", "")
    sect_pe  = SECTOR_PE.get(sector, SECTOR_PE["default"])

    # SIG-03 FIX: FMCG/Consumer names sustain higher PEs — use wider ceiling
    if sector in HIGH_PE_SECTORS:
        sustainable_pe = min(sect_pe * 1.5, 35 + max(0, growth) / 2)
    else:
        sustainable_pe = min(sect_pe * 1.2, 25 + max(0, growth) / 2)
    sustainable_pe = max(12.0, min(sustainable_pe, 80.0))

    if pe and pe > 0 and growth is not None:
        g = max(0, growth) / 100

        # BUG-03 FIX: proper 1yr formula: forward EPS × sustainable PE
        fwd_eps_1yr = (1 / pe) * (1 + g)
        result["target_1yr"] = round(cmp * fwd_eps_1yr * sustainable_pe, 2)

        # 2yr: two years of EPS growth + PE re-rating
        fwd_eps_2yr = (1 / pe) * ((1 + g) ** 2)
        result["target_2yr"] = round(cmp * fwd_eps_2yr * sustainable_pe, 2)
        result["fair_value"]  = result["target_2yr"]

        # BUG-02 FIX: exponent 3.0 (was 2.5)
        fwd_eps_3yr = (1 / pe) * ((1 + g) ** 3.0)
        result["target_3yr"] = round(cmp * fwd_eps_3yr * sustainable_pe, 2)

    # Blend with analyst target if credible (≥3 analysts)
    analyst_tgt = fund.get("analyst_target")
    analyst_cnt = fund.get("analyst_count") or 0
    if analyst_tgt and analyst_cnt >= 3 and analyst_tgt > cmp * 0.8:
        if result["target_2yr"]:
            result["target_2yr"] = round(result["target_2yr"] * 0.6 + analyst_tgt * 0.4, 2)
        else:
            result["target_2yr"] = round(analyst_tgt, 2)

    # Cross-check with pattern measured move
    if pattern and pattern.get("bz") and pattern.get("bottom"):
        bz    = pattern["bz"]
        depth = bz - pattern["bottom"]
        pat_t = bz + depth
        if result["target_2yr"]:
            result["target_2yr"] = round(max(result["target_2yr"], pat_t), 2)
        else:
            result["target_2yr"] = round(pat_t, 2)

    # Derived
    if result["target_2yr"] and result["target_2yr"] > cmp:
        result["upside_potential_pct"] = round((result["target_2yr"] - cmp) / cmp * 100, 1)
        result["expected_cagr_2yr"]    = round(((result["target_2yr"] / cmp) ** 0.5 - 1) * 100, 1)

    return result


# ═══════════════════════════════════════════════════════════════════════
# HARD FILTER
# ═══════════════════════════════════════════════════════════════════════
def passes_hard_filter(fund: dict, cmp: float) -> tuple[bool, str]:
    """
    SIG-11 FIX: interest coverage added.
    """
    mc = fund.get("market_cap")
    if mc:
        mc_cr = mc / 1e7
        if mc_cr < MIN_MARKET_CAP_CR:
            return False, f"Market cap ₹{mc_cr:.0f}Cr < minimum"

    roe = fund.get("roe_screener") or fund.get("roe")
    if roe is not None and roe < HARD_FILTERS["min_roe"]:
        return False, f"ROE {roe:.1f}% < {HARD_FILTERS['min_roe']}%"

    de = fund.get("debt_to_equity_ratio")
    if de is not None and de > HARD_FILTERS["max_debt_equity"]:
        return False, f"D/E {de:.2f} > {HARD_FILTERS['max_debt_equity']}"

    rev = fund.get("sales_3yr_cagr") or fund.get("revenue_growth_ttm")
    if rev is not None and rev < HARD_FILTERS["min_revenue_growth"]:
        return False, f"Revenue growth {rev:.1f}% < {HARD_FILTERS['min_revenue_growth']}%"

    pe = fund.get("pe_ratio") or fund.get("pe_screener")
    if pe is not None and pe > HARD_FILTERS["max_pe"]:
        return False, f"PE {pe:.0f} > {HARD_FILTERS['max_pe']}"

    # SIG-11 FIX: interest coverage
    icr = fund.get("interest_coverage")
    de_val = fund.get("debt_to_equity_ratio") or 0
    if icr is not None and de_val > 0.3 and icr < HARD_FILTERS["min_interest_coverage"]:
        return False, f"Interest coverage {icr:.1f}× < {HARD_FILTERS['min_interest_coverage']}×"

    return True, "OK"


# ═══════════════════════════════════════════════════════════════════════
# WHY BUY / KEY RISKS
# ═══════════════════════════════════════════════════════════════════════
def build_why_buy(f_r: list, t_r: list, v_r: list, m_r: list,
                  fund: dict, targets: dict) -> str:
    all_r = f_r + v_r + t_r + m_r
    seen  = set(); unique = []
    for r in all_r:
        if r not in seen:
            seen.add(r); unique.append(r)
    cagr = targets.get("expected_cagr_2yr")
    if cagr and cagr > 0:
        unique.insert(0, f"Expected ~{cagr:.0f}% CAGR (2yr)")
    return " | ".join(unique[:6])


def build_key_risks(fund: dict, pattern: dict | None) -> str:
    risks = []
    de = fund.get("debt_to_equity_ratio")
    if de and de > 1.0:
        risks.append(f"High D/E {de:.1f}")
    pledge = fund.get("promoter_pledge", 0) or 0
    if pledge > 20:
        risks.append(f"Promoter pledge {pledge:.0f}%")
    de_trend = fund.get("de_trend")
    if de_trend and de_trend > 0.05:
        risks.append("Rising debt ↑")
    pe = fund.get("pe_ratio") or fund.get("pe_screener")
    if pe and pe > 50:
        risks.append(f"High PE {pe:.0f}")
    opm_trend = fund.get("opm_trend")
    if opm_trend and opm_trend < -0.5:
        risks.append("Margin compression ↓")
    ne = fund.get("next_earnings")
    if ne:
        risks.append(f"Earnings due {ne}")
    if pattern and pattern.get("status") == "Forming":
        risks.append("Pattern still forming (not broken out)")
    return " | ".join(risks[:4]) if risks else "None identified"


# ═══════════════════════════════════════════════════════════════════════
# SCAN ONE STOCK
# ═══════════════════════════════════════════════════════════════════════
def scan_one(sym: str, n50_mo: pd.DataFrame | None,
             market_ctx: dict) -> dict | None:
    mo_df = read_cache(sym, "1mo")
    wk_df = read_cache(sym, "1wk")
    d_df  = read_cache(sym, "1d")
    fund  = read_fund(sym)

    if mo_df is None or len(mo_df) < MIN_MONTHLY_BARS:
        return None
    if not fund:
        return None

    c   = mo_df["Close"].values.astype(float)
    # SIG-05 FIX: extract High/Low for pattern detectors
    h_arr  = mo_df["High"].values.astype(float)   if "High"   in mo_df.columns else None
    lo_arr = mo_df["Low"].values.astype(float)    if "Low"    in mo_df.columns else None
    v      = mo_df["Volume"].values.astype(float) if "Volume" in mo_df.columns else None
    cmp    = round(float(c[-1]), 2)

    passes, reason = passes_hard_filter(fund, cmp)
    if not passes:
        log.debug(f"{sym}: filtered — {reason}")
        return None

    # ── Monthly pattern detection ─────────────────────────────────────────────
    detectors = [
        det_monthly_cup, det_monthly_vcp, det_monthly_flat_base,
        det_monthly_stage2, det_monthly_asc_triangle, det_monthly_double_bottom,
    ]
    found = []
    for det in detectors:
        try:
            res = det(c, v, h_arr, lo_arr)
            if res is not None:
                found.append(res)
        except Exception:
            pass

    best_pattern = max(found, key=lambda p: p["quality"]) if found else None
    converging   = ("+".join(sorted({p["pattern"] for p in found}))
                    if len(found) > 1 else None)

    # ── Sector context ────────────────────────────────────────────────────────
    sector        = fund.get("sector", "")
    sector_trend  = get_sector_trend(sector)
    stage_age     = get_sector_stage_age(sector)

    # ── Multi-timeframe ───────────────────────────────────────────────────────
    wk = analyze_weekly(wk_df)
    d  = analyze_daily(d_df)

    # ── Scoring ───────────────────────────────────────────────────────────────
    f_score, f_reasons = score_fundamentals(fund)
    t_score, t_reasons = score_technical(best_pattern, wk, d, market_ctx,
                                          sector_trend, stage_age)
    v_score, v_reasons = score_valuation(fund, cmp)
    m_score, m_reasons = score_momentum(mo_df, n50_mo, fund)
    total              = round(f_score + t_score + v_score + m_score, 1)

    # ── Grade ─────────────────────────────────────────────────────────────────
    if total >= 80:   grade = "STRONG BUY ⭐⭐⭐⭐⭐"
    elif total >= 70: grade = "BUY ⭐⭐⭐⭐"
    elif total >= 60: grade = "ACCUMULATE ⭐⭐⭐"
    elif total >= 50: grade = "WATCH ⭐⭐"
    elif total >= 35: grade = "MONITOR ⭐"
    else:             grade = "SKIP"

    targets   = calc_lt_targets(cmp, fund, best_pattern)
    why_buy   = build_why_buy(f_reasons, t_reasons, v_reasons, m_reasons, fund, targets)
    key_risks = build_key_risks(fund, best_pattern)

    # ── Horizon / risk ────────────────────────────────────────────────────────
    de        = fund.get("debt_to_equity_ratio") or 0
    if total >= 75 and de < 0.3:
        horizon = "1-3 years"; risk = "Low-Medium"
    elif total >= 65:
        horizon = "2-3 years"; risk = "Medium"
    else:
        horizon = "3-4 years"; risk = "Medium-High"

    # ── SIG-12 FIX: SEBI FY25 market cap classification ──────────────────────
    mc    = fund.get("market_cap") or 0
    mc_cr = mc / 1e7
    if mc_cr >= 30000:    cap_class = "Large-Cap"
    elif mc_cr >= 8000:   cap_class = "Mid-Cap"
    elif mc_cr >= 500:    cap_class = "Small-Cap"
    else:                 cap_class = "Micro-Cap"

    # ── RS metrics ────────────────────────────────────────────────────────────
    rs_1yr = calc_rs_vs_nifty(mo_df, n50_mo, 1) if n50_mo is not None else None
    rs_3yr = calc_rs_vs_nifty(mo_df, n50_mo, 3) if n50_mo is not None else None

    # ── 52-week ───────────────────────────────────────────────────────────────
    wkh      = fund.get("fifty_two_week_high")
    wkl      = fund.get("fifty_two_week_low")
    dist_52h = round((wkh - cmp) / wkh * 100, 1) if wkh and wkh > 0 else None
    dist_52l = round((cmp - wkl) / wkl * 100, 1) if wkl and wkl > 0 else None

    # ── Data quality ──────────────────────────────────────────────────────────
    sources    = fund.get("_sources", [])
    key_fields = [fund.get(f) for f in ("roe","roce","debt_to_equity_ratio","promoter_holding",
                                        "sales_3yr_cagr","profit_3yr_cagr","pe_ratio","pb_ratio",
                                        "operating_margin","free_cashflow")]
    data_quality = round(sum(1 for f in key_fields if f is not None) / len(key_fields) * 100)

    # ── Piotroski score for output ────────────────────────────────────────────
    piotroski_f, _ = compute_piotroski(fund)
    eps_con, eps_desc = compute_eps_consistency(fund)

    return {
        # ── Identification
        "Stock":                       sym.replace(".NS", ""),
        "Company_Name":                fund.get("long_name", ""),
        "Sector":                      sector,
        "Industry":                    fund.get("industry", ""),
        "Market_Cap_Cr":               round(mc_cr, 0),
        "Cap_Class":                   cap_class,
        # ── Pattern
        "Monthly_Pattern":             best_pattern["pattern"] if best_pattern else "None",
        "Monthly_Pattern_Status":      best_pattern["status"]  if best_pattern else "",
        "Monthly_Pattern_Quality":     best_pattern["quality"] if best_pattern else 0,
        "Monthly_Pattern_Duration_Mo": best_pattern["duration_months"] if best_pattern else 0,
        "Monthly_Pattern_Details":     best_pattern["details"] if best_pattern else "",
        "Converging_Signals":          converging or "",
        # ── Multi-timeframe
        "Weekly_Trend":                wk["trend"],
        "Weekly_Above_40MA":           str(wk["above_40wma"]),
        "Weekly_Volume_Expanding":     str(wk.get("vol_expanding", False)),
        "Daily_Setup":                 d["setup"],
        "Daily_Above_50MA":            str(d["above_50dma"]),
        "Sector_Trend":                sector_trend,
        "Sector_Stage2_Age_Mo":        stage_age,
        "Index_Regime":                market_ctx.get("regime", ""),
        "Nifty_Monthly_Trend":         market_ctx.get("nifty_trend_mo", ""),
        # ── Price levels
        "CMP":                         cmp,
        "Breakout_Zone":               best_pattern["bz"]     if best_pattern else "",
        "Pattern_Bottom":              best_pattern["bottom"] if best_pattern else "",
        "Stop_Loss":                   targets["stop_loss"],
        "Target_1yr":                  targets["target_1yr"],
        "Target_2yr":                  targets["target_2yr"],
        "Target_3yr":                  targets["target_3yr"],
        "Fair_Value_Estimate":         targets["fair_value"],
        "Upside_Potential_Pct":        targets["upside_potential_pct"],
        "Expected_CAGR_2yr_Pct":       targets["expected_cagr_2yr"],
        # ── Valuation
        "PE_Ratio":                    _fmt(fund.get("pe_ratio") or fund.get("pe_screener")),
        "Forward_PE":                  _fmt(fund.get("forward_pe")),
        "PB_Ratio":                    _fmt(fund.get("pb_ratio")),
        "PEG_Ratio":                   _fmt(fund.get("_peg")),
        "EV_EBITDA":                   _fmt(fund.get("ev_ebitda")),
        "Dividend_Yield_Pct":          _fmt(fund.get("dividend_yield") or fund.get("dividend_yield_screener")),
        "Dividend_Growth_3yr_Pct":     _fmt(fund.get("dividend_growth_3yr")),
        "Analyst_Target":              _fmt(fund.get("analyst_target")),
        "Analyst_Count":               fund.get("analyst_count", ""),
        "Analyst_Recommendation":      fund.get("recommendation", ""),
        # ── Profitability
        "ROE_Pct":                     _fmt(fund.get("roe_screener") or fund.get("roe")),
        "ROCE_Pct":                    _fmt(fund.get("roce")),
        "ROCE_Trend":                  _fmt(fund.get("roce_trend")),
        "Operating_Margin_Pct":        _fmt(fund.get("operating_margin_screener") or fund.get("operating_margin")),
        "OPM_Trend":                   _fmt(fund.get("opm_trend")),
        "Net_Margin_Pct":              _fmt(fund.get("net_margin")),
        "Return_on_Assets_Pct":        _fmt(fund.get("roa")),
        # ── Growth
        "Revenue_Growth_TTM_Pct":      _fmt(fund.get("sales_growth_ttm") or fund.get("revenue_growth_ttm")),
        "Revenue_Growth_3yr_CAGR":     _fmt(fund.get("sales_3yr_cagr")),
        "Revenue_Growth_5yr_CAGR":     _fmt(fund.get("sales_5yr_cagr")),
        "Profit_Growth_TTM_Pct":       _fmt(fund.get("profit_growth_ttm") or fund.get("earnings_growth_ttm")),
        "Profit_Growth_3yr_CAGR":      _fmt(fund.get("profit_3yr_cagr")),
        "Profit_Growth_5yr_CAGR":      _fmt(fund.get("profit_5yr_cagr")),
        "EPS_Growth_Quarterly_Pct":    _fmt(fund.get("earnings_quarterly_growth")),
        "EPS_Consistency_Score":       round(eps_con, 2),
        "EPS_Consistency_Desc":        eps_desc,
        # ── Balance sheet
        "Debt_to_Equity":              _fmt(fund.get("debt_to_equity_ratio")),
        "DE_Trend":                    _fmt(fund.get("de_trend")),
        "Current_Ratio":               _fmt(fund.get("current_ratio")),
        "Interest_Coverage":           _fmt(fund.get("interest_coverage")),
        "Free_Cashflow_Cr":            _fmt(_to_cr(fund.get("free_cashflow"))),
        "FCF_Yield_Pct":               _fmt(round(fund.get("free_cashflow", 0) / max(mc, 1) * 100, 2)
                                           if fund.get("free_cashflow") and mc > 0 else None),
        # ── Ownership
        "Promoter_Holding_Pct":        _fmt(fund.get("promoter_holding")),
        "Promoter_Change_QoQ":         _fmt(fund.get("promoter_delta")),
        "Promoter_Pledge_Pct":         _fmt(fund.get("promoter_pledge")),
        "FII_Holding_Pct":             _fmt(fund.get("fii_holding") or fund.get("held_pct_institutions")),
        "FII_Change_QoQ":              _fmt(fund.get("fii_delta")),
        "DII_Holding_Pct":             _fmt(fund.get("dii_holding")),
        "DII_Change_QoQ":              _fmt(fund.get("dii_delta")),
        # ── Technical
        "52wk_High":                   _fmt(wkh),
        "52wk_Low":                    _fmt(wkl),
        "Dist_From_52wk_High_Pct":     dist_52h,
        "Dist_From_52wk_Low_Pct":      dist_52l,
        "RS_vs_Nifty_1yr_Pct":         _fmt(rs_1yr),
        "RS_vs_Nifty_3yr_Pct":         _fmt(rs_3yr),
        "Monthly_Vol_Surge":           _fmt(best_pattern["vs"] if best_pattern else None),
        "Beta":                        _fmt(fund.get("beta")),
        # ── Quality composite
        "Piotroski_F_Score":           piotroski_f,
        # ── Scores
        "Fundamental_Score_40":        f_score,
        "Technical_Score_30":          t_score,
        "Valuation_Score_20":          v_score,
        "Momentum_Score_10":           m_score,
        "Overall_Score_100":           total,
        "Rating_Grade":                grade,
        # ── Analysis
        "Why_Buy":                     why_buy,
        "Key_Risks":                   key_risks,
        "Investment_Horizon":          horizon,
        "Risk_Level":                  risk,
        # ── Metadata
        "Data_Sources":                ", ".join(sources) if sources else "yfinance",
        "Data_Quality_Pct":            data_quality,
        "Monthly_Bars":                len(mo_df),
        "Next_Earnings":               fund.get("next_earnings", ""),
        "Scan_Date":                   str(_today()),
    }


# ═══════════════════════════════════════════════════════════════════════
# SCAN HISTORY
# ═══════════════════════════════════════════════════════════════════════
def update_history(results: list[dict]):
    con = _get_db()
    with _db_lock:
        for r in results:
            stock = r["Stock"]
            price = r["CMP"]
            score = r["Overall_Score_100"]
            existing = con.execute(
                "SELECT first_seen_date,first_price,first_score,times_scanned,pattern_history "
                "FROM stock_history WHERE stock=?", (stock,)
            ).fetchone()
            if existing:
                pats = json.loads(existing[4] or "[]")
                pat  = r.get("Monthly_Pattern", "")
                if pat and pat not in pats:
                    pats.append(pat)
                con.execute(
                    "UPDATE stock_history SET last_seen_date=?,last_price=?,last_score=?,"
                    "times_scanned=times_scanned+1,pattern_history=? WHERE stock=?",
                    (str(_today()), price, score, json.dumps(pats), stock))
            else:
                con.execute(
                    "INSERT INTO stock_history "
                    "(stock,first_seen_date,first_price,first_score,"
                    "last_seen_date,last_price,last_score,times_scanned,pattern_history) "
                    "VALUES (?,?,?,?,?,?,?,1,?)",
                    (stock, str(_today()), price, score,
                     str(_today()), price, score,
                     json.dumps([r.get("Monthly_Pattern", "")])))
        con.commit()


def load_history() -> dict:
    try:
        con  = _get_db()
        rows = con.execute("SELECT * FROM stock_history").fetchall()
        cols = [d[0] for d in con.execute(
            "SELECT * FROM stock_history LIMIT 0").description]
        return {r[0]: dict(zip(cols, r)) for r in rows}
    except Exception:
        return {}


def enrich_with_history(results: list[dict], history: dict) -> list[dict]:
    for r in results:
        stock = r["Stock"]
        h     = history.get(stock)
        if h:
            r["First_Seen_Date"] = h.get("first_seen_date", "")
            fsd = h.get("first_seen_date")
            r["Days_On_Watchlist"] = (
                (_today() - datetime.strptime(fsd, "%Y-%m-%d").date()).days
                if fsd else ""
            )
            fp = h.get("first_price")
            r["Price_Change_Since_First_Pct"] = (
                round((r["CMP"] - fp) / fp * 100, 1) if fp and fp > 0 else ""
            )
            r["Times_Scanned"]     = h.get("times_scanned", 1)
            fs = h.get("first_score")
            r["Score_vs_First"]    = round(r["Overall_Score_100"] - fs, 1) if fs else ""
        else:
            r["First_Seen_Date"]              = str(_today())
            r["Days_On_Watchlist"]            = 0
            r["Price_Change_Since_First_Pct"] = ""
            r["Times_Scanned"]                = 1
            r["Score_vs_First"]               = ""
    return results


# ═══════════════════════════════════════════════════════════════════════
# SAVE RESULTS  (GAP-10: max-per-sector deduplication)
# ═══════════════════════════════════════════════════════════════════════
def apply_sector_cap(results: list[dict], max_per_sector: int) -> list[dict]:
    """
    GAP-10 FIX: cap number of stocks per sector to ensure diversification.
    Keeps top N per sector by score (already sorted desc).
    """
    if max_per_sector <= 0:
        return results
    sector_count: dict[str, int] = {}
    filtered = []
    for r in results:
        s = r.get("Sector", "Unknown")
        sector_count[s] = sector_count.get(s, 0) + 1
        if sector_count[s] <= max_per_sector:
            filtered.append(r)
    removed = len(results) - len(filtered)
    if removed > 0:
        log.info(f"Sector cap (max {max_per_sector}/sector): removed {removed} stocks")
    return filtered


def save_results(results: list[dict], scan_id: str, max_per_sector: int = 0):
    if not results:
        log.warning("No results to save"); return

    # Apply sector cap if requested
    results = apply_sector_cap(results, max_per_sector)

    con = _get_db()
    with _db_lock:
        for r in results:
            r_native = _to_native(r)  # Convert numpy types to native Python
            con.execute(
                "INSERT OR REPLACE INTO scan_results "
                "(scan_id,scan_date,stock,score,grade,result_json) VALUES (?,?,?,?,?,?)",
                (scan_id, str(_today()), r["Stock"],
                 r["Overall_Score_100"], r["Rating_Grade"], json.dumps(r_native)))
        con.commit()

    ts  = _now().strftime("%Y%m%d_%H%M")
    df  = pd.DataFrame(results)

    all_csv   = os.path.join(OUTPUT_DIR, f"invest_ALL_{ts}.csv")
    buy_csv   = os.path.join(OUTPUT_DIR, f"invest_BUY_{ts}.csv")
    watch_csv = os.path.join(OUTPUT_DIR, f"invest_WATCH_{ts}.csv")

    df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    buy_df   = df[df["Rating_Grade"].str.startswith("STRONG BUY") |
                  df["Rating_Grade"].str.startswith("BUY")]
    buy_df.to_csv(buy_csv, index=False, encoding="utf-8-sig")

    watch_df = df[df["Rating_Grade"].str.startswith("ACCUMULATE") |
                  df["Rating_Grade"].str.startswith("WATCH")]
    watch_df.to_csv(watch_csv, index=False, encoding="utf-8-sig")

    log.info(f"\n{'='*60}")
    log.info(f"SCAN: {len(results)} rated | STRONG BUY+BUY={len(buy_df)} | "
             f"ACCUMULATE+WATCH={len(watch_df)}")
    log.info(f"ALL CSV   → {os.path.basename(all_csv)}")
    log.info(f"BUY CSV   → {os.path.basename(buy_csv)}")
    log.info(f"WATCH CSV → {os.path.basename(watch_csv)}")
    log.info(f"{'='*60}")

    if len(buy_df) > 0:
        cols = ["Stock","Cap_Class","Monthly_Pattern","CMP","Target_2yr",
                "Expected_CAGR_2yr_Pct","Piotroski_F_Score","Overall_Score_100",
                "Rating_Grade","Why_Buy"]
        print("\n─── TOP BUY CANDIDATES ───")
        print(buy_df[[c for c in cols if c in buy_df.columns]].head(20).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════
# SCAN RUNNERS
# ═══════════════════════════════════════════════════════════════════════
def run_full_scan(stocks: list[str], min_score: float = 35.0) -> list[dict]:
    log.info(f"=== INVESTMENT SCAN: {len(stocks)} stocks | min_score={min_score} ===")
    t0  = time.time()
    mkt = load_market_context()
    log.info(f"Market: {mkt['regime']} | Nifty-Mo: {mkt['nifty_trend_mo']} | "
             f"Nifty-Wk: {mkt['nifty_trend_wk']} | VIX: {mkt.get('vix','N/A')}")

    n50_mo = read_cache(NIFTY50_SYM, "1mo")
    if n50_mo is None:
        log.warning("Nifty50 monthly not cached — RS vs Nifty won't be computed")

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
            except Exception:
                pass

    all_results.sort(key=lambda x: -x["Overall_Score_100"])
    log.info(f"Scan done: {time.time()-t0:.0f}s | {len(all_results)} ≥ {min_score}")
    return all_results


def run_watchlist_scan(min_score: float = 35.0) -> list[dict]:
    """
    SIG-17 FIX: min_score is now actually applied to filter results.
    """
    history = load_history()
    if not history:
        log.warning("No history — run --scan first"); return []

    stocks = [s + ".NS" for s in history.keys()]
    log.info(f"Watchlist rescan: {len(stocks)} stocks")

    mkt    = load_market_context()
    n50_mo = read_cache(NIFTY50_SYM, "1mo")
    results = []
    for sym in stocks:
        try:
            r = scan_one(sym, n50_mo, mkt)
            # SIG-17 FIX: apply min_score filter
            if r and r["Overall_Score_100"] >= min_score:
                results.append(r)
        except Exception:
            pass

    results.sort(key=lambda x: -x["Overall_Score_100"])
    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="NSE Long-Term Investment Scanner v2")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scan",      action="store_true")
    mode.add_argument("--watchlist", action="store_true")
    mode.add_argument("--history",   action="store_true")
    mode.add_argument("--test",      action="store_true")

    ap.add_argument("--min-score",      type=float, default=35.0)
    ap.add_argument("--limit",          type=int,   default=0)
    # GAP-10 FIX: max stocks per sector for output diversification
    ap.add_argument("--max-per-sector", type=int,   default=0,
                    help="Cap results per sector (0 = unlimited). E.g. 3 = top 3 per sector")
    args = ap.parse_args()

    _get_db()

    # ── History ────────────────────────────────────────────────────────────────
    if args.history:
        hist = load_history()
        if not hist:
            print("No history yet. Run --scan first."); return
        print(f"\n{'Stock':<15} {'First':<12} {'₹First':>10} {'₹Now':>10} "
              f"{'Δ%':>8} {'Score':>7} {'Times':>6}")
        print("-" * 72)
        for r in sorted(hist.values(), key=lambda x: x.get("last_score", 0), reverse=True):
            fp  = r.get("first_price", 0) or 0
            lp  = r.get("last_price",  0) or 0
            chg = round((lp - fp) / fp * 100, 1) if fp > 0 else 0
            print(f"{r['stock']:<15} {r.get('first_seen_date',''):<12} "
                  f"₹{fp:>9.2f} ₹{lp:>9.2f} {chg:>+8.1f}% "
                  f"{r.get('last_score',0):>7.1f} {r.get('times_scanned',0):>6}")
        return

    # ── Test mode  (GAP-12 FIX: mix of large + mid + small caps) ──────────────
    if args.test:
        # GAP-12 FIX: includes mid/small-cap compounders — the scanner's actual target
        TEST_STOCKS = [
            # Large-caps
            "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC",
            "BHARTIARTL","KOTAKBANK","LT","BAJFINANCE","ASIANPAINT","TITAN",
            # Mid-caps (₹8k–30k Cr) — where compounders are found early
            "PAGEIND","DMART","PIIND","PERSISTENT","COFORGE","LTTS","HAPPSTMNDS",
            "FINPIPE","APLAPOLLO","KPRMILL","GARFIBRES","AAVAS","CREDITACC",
            "METROPOLIS","VIJAYA","WELCORP","CRAFTSMAN","BIKAJI","SAPPHIRE",
            # Small-caps (₹500–8k Cr) — future compounders
            "NEWGEN","CAMPUS","IDEAFORGE","DODLA","GLOBUSS","RKFORGE","LXCHEM",
            "RATEGAIN","HOMEFIRST","PPLPHARMA","SIGACHI","TATTECH","MAZDOCK",
        ]
        stocks  = [s + ".NS" for s in TEST_STOCKS if "." not in s]
        stocks += [s for s in TEST_STOCKS if s.endswith(".NS")]
        log.info(f"Test mode: {len(stocks)} stocks (large + mid + small cap mix)")

        mkt    = load_market_context()
        n50_mo = read_cache(NIFTY50_SYM, "1mo")
        hist   = load_history()

        results = []
        for sym in stocks:
            try:
                r = scan_one(sym, n50_mo, mkt)
                if r:
                    results.append(r)
            except Exception as e:
                log.debug(f"{sym}: {e}")

        results.sort(key=lambda x: -x["Overall_Score_100"])
        results = enrich_with_history(results, hist)
        update_history(results)
        save_results(results, f"test_{_today()}", args.max_per_sector)
        return

    # ── Watchlist rescan ───────────────────────────────────────────────────────
    if args.watchlist:
        hist    = load_history()
        results = run_watchlist_scan(args.min_score)
        results = enrich_with_history(results, hist)
        update_history(results)
        save_results(results, f"watchlist_{_today()}", args.max_per_sector)
        return

    # ── Full scan ──────────────────────────────────────────────────────────────
    if args.scan:
        stocks = load_universe_from_cache()
        if not stocks:
            log.error("No stocks in cache. Run: python data_updater.py --bootstrap")
            sys.exit(1)
        if args.limit > 0:
            stocks = stocks[:args.limit]
        hist    = load_history()
        results = run_full_scan(stocks, args.min_score)
        results = enrich_with_history(results, hist)
        update_history(results)
        save_results(results, f"scan_{_today()}", args.max_per_sector)


if __name__ == "__main__":
    main()
