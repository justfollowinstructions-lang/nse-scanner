#!/usr/bin/env python3
"""
NSE Long-Term Investment Data Updater  (v2 — fully audited)
=============================================================
All audit issues addressed. Change log vs v1:

BUG-10  D/E always divided by 100 (yfinance always returns %)
BUG-11  FUND_REFRESH_DAYS raised to 30 (was 7 — caused full re-download every bi-weekly run)
BUG-12  Universe pre-filtered to EQ series with column guard
BUG-14  _pct_ratio() for yfinance growth/margin fields (always 0-1 decimal, never guess)
BUG-15  Cache key in scanner.yml changed to monthly (see that file)
SIG-09  run_update stale list now respects 30-day window correctly
SIG-14  load_universe has strict SERIES column guard + hard-fail if missing
SIG-18  _cagr handles turnarounds (negative base year → capped positive signal)
GAP-01  _parse_shareholding stores last 4 quarters of FII / DII for trend detection
GAP-02  _parse_ratios_history stores historical D/E list for slope computation
GAP-03  _parse_pl_table stores profit_history list for EPS consistency in scanner
GAP-04  _parse_ratios_history stores historical ROCE list for trend
GAP-07  fetch_yf_fundamentals fetches dividend history → dividend_growth_3yr
GAP-08  promoter_history stored (4 quarters) for trend detection
GAP-09  screener data freshness validation vs yfinance market cap (>50% diff → warning)
MIN-02  warnings scoped to yfinance/pandas only, never silence all
MIN-03  per-thread read connections via threading.local() — no lock contention on reads
MIN-06  cache_meta.bar_count stores actual total bars in DB, not batch count
"""

import os, sys, json, time, sqlite3, argparse, logging, threading
from datetime import date, datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
import warnings

# MIN-02 FIX: targeted suppression only
warnings.filterwarnings("ignore", category=FutureWarning,      module="yfinance")
warnings.filterwarnings("ignore", category=FutureWarning,      module="pandas")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="yfinance")
warnings.filterwarnings("ignore", message=".*auto_adjust.*")
warnings.filterwarnings("ignore", message=".*Timezone.*")

import yfinance as yf
import pandas as pd
import numpy as np
import requests

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

# ═══════════════════════════════════════════════════════════════════════
# PATHS & LOGGING
# ═══════════════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "invest_cache.db")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(LOG_DIR, f"updater_{date.today()}.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("data_updater")

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
_IST = timezone(timedelta(hours=5, minutes=30))
def _now():   return datetime.now(_IST)
def _today(): return _now().date()

MAX_WORKERS       = 4
DL_RETRIES        = 3
DL_BACKOFF        = 4.0
# BUG-11 / SIG-09 FIX: 7 → 30 days.  The bi-weekly scan (every 15 days)
# now finds all fundamentals fresh (15 < 30) and skips re-download.
FUND_REFRESH_DAYS = 30
SCREENER_DELAY    = 2.5    # seconds between screener.in requests
MIN_MARKET_CAP_CR = 200

TF_CONFIG = {
    "1mo": "max",
    "1wk": "10y",
    "1d":  "2y",
}

INDICES = {
    "NIFTY50":          "^NSEI",
    "NIFTYBANK":        "^NSEBANK",
    "NIFTYMIDCAP50":    "^NSEMDCP50",
    "INDIAVIX":         "^INDIAVIX",
    "NIFTYIT":          "^CNXIT",
    "NIFTYPHARMA":      "^CNXPHARMA",
    "NIFTYAUTO":        "^CNXAUTO",
    "NIFTYFMCG":        "^CNXFMCG",
    "NIFTYMETAL":       "^CNXMETAL",
    "NIFTYINFRA":       "^CNXINFRA",
    "NIFTYENERGY":      "^CNXENERGY",
    "NIFTYREALTY":      "^CNXREALTY",
    "NIFTYCONSUMPTION": "^CNXCONSUM",
    "NIFTYPSUBANK":     "^CNXPSUBANK",
}

# ═══════════════════════════════════════════════════════════════════════
# DATABASE  (write connection + per-thread read connections)
# ═══════════════════════════════════════════════════════════════════════
_db_lock = threading.Lock()
_db_con  = None
_local   = threading.local()   # MIN-03 FIX


def _get_db() -> sqlite3.Connection:
    """Single write-capable connection."""
    global _db_con
    with _db_lock:
        if _db_con is None:
            _db_con = sqlite3.connect(CACHE_PATH, check_same_thread=False)
            _db_con.execute("PRAGMA journal_mode=WAL")
            _db_con.execute("PRAGMA synchronous=NORMAL")
            _db_con.execute("PRAGMA cache_size=-131072")
            _db_con.executescript("""
                CREATE TABLE IF NOT EXISTS price_cache (
                    stock TEXT NOT NULL, tf TEXT NOT NULL, date TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL NOT NULL, volume REAL,
                    PRIMARY KEY (stock, tf, date)
                );
                CREATE TABLE IF NOT EXISTS cache_meta (
                    stock TEXT NOT NULL, tf TEXT NOT NULL,
                    last_date TEXT, last_updated TEXT, bar_count INTEGER,
                    PRIMARY KEY (stock, tf)
                );
                CREATE TABLE IF NOT EXISTS fund_cache (
                    stock TEXT PRIMARY KEY, fund_json TEXT, updated_date TEXT
                );
                CREATE TABLE IF NOT EXISTS index_cache (
                    symbol TEXT NOT NULL, tf TEXT NOT NULL, date TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL NOT NULL, volume REAL,
                    PRIMARY KEY (symbol, tf, date)
                );
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
                CREATE INDEX IF NOT EXISTS idx_pc_stock_tf ON price_cache(stock, tf);
                CREATE INDEX IF NOT EXISTS idx_pc_date     ON price_cache(tf, date);
                CREATE INDEX IF NOT EXISTS idx_meta        ON cache_meta(stock, tf);
            """)
            _db_con.commit()
        return _db_con


def _get_read_con() -> sqlite3.Connection:
    """MIN-03 FIX: per-thread read-only connection — zero lock contention."""
    if not hasattr(_local, "con"):
        con = sqlite3.connect(CACHE_PATH, check_same_thread=True)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA query_only=ON")
        _local.con = con
    return _local.con


# ═══════════════════════════════════════════════════════════════════════
# YAHOO FINANCE SESSION
# ═══════════════════════════════════════════════════════════════════════
_YF_SESSION = None
_YF_LOCK    = threading.Lock()


def _build_session():
    try:
        from curl_cffi import requests as _cr
        sess = _cr.Session(impersonate="chrome110")
        sess.get("https://finance.yahoo.com", timeout=15)
        log.info("curl_cffi Chrome session ready")
        return sess
    except ImportError:
        log.warning("curl_cffi not installed — may get 401 on CI. pip install curl_cffi")
        return None
    except Exception as e:
        log.warning(f"Session build: {e}"); return None


def _get_session():
    global _YF_SESSION
    with _YF_LOCK:
        if _YF_SESSION is None:
            _YF_SESSION = _build_session()
        return _YF_SESSION


def _reset_session():
    global _YF_SESSION
    with _YF_LOCK:
        _YF_SESSION = None


# ═══════════════════════════════════════════════════════════════════════
# OHLCV DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════
def _yf_download(sym: str, **kwargs) -> pd.DataFrame | None:
    """Shared download with retry / rate-limit handling."""
    for attempt in range(DL_RETRIES):
        try:
            sess = _get_session()
            kw   = {"session": sess} if sess else {}
            kw.update(kwargs)
            import contextlib, io as _io
            with contextlib.redirect_stderr(_io.StringIO()):
                df = yf.download(sym, auto_adjust=True, progress=False, timeout=30, **kw)
            if df is None or len(df) == 0:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close"])
            df.columns = [c.capitalize() if c.lower() in
                          ("open","high","low","close","volume") else c for c in df.columns]
            return df if len(df) >= 2 else None
        except Exception as e:
            msg = str(e)
            if "401" in msg or "Crumb" in msg or "Unauthorized" in msg:
                log.warning(f"401 {sym} attempt {attempt+1}"); _reset_session(); time.sleep(10)
            elif "429" in msg or "rate limit" in msg.lower():
                time.sleep(20 * (attempt + 1))
            elif "delisted" in msg.lower() or "no price data" in msg.lower():
                return None
            elif attempt < DL_RETRIES - 1:
                time.sleep(DL_BACKOFF * (attempt + 1))
    return None


def dl_ohlcv(sym: str, tf: str = "1mo", period: str = "max") -> pd.DataFrame | None:
    return _yf_download(sym, period=period, interval=tf)


def dl_since(sym: str, tf: str, since: str) -> pd.DataFrame | None:
    return _yf_download(sym, start=since, interval=tf)


# ═══════════════════════════════════════════════════════════════════════
# CACHE READ / WRITE
# ═══════════════════════════════════════════════════════════════════════
def _safe_float(row, col: str) -> float:
    v = row.get(col, row.get(col.lower(), 0))
    try:
        f = float(v); return 0.0 if f != f else f
    except Exception:
        return 0.0


def write_cache(stock: str, tf: str, df: pd.DataFrame) -> int:
    if df is None or len(df) == 0:
        return 0
    rows = []
    for idx, row in df.iterrows():
        d = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)[:10]
        rows.append((stock, tf, d,
                     _safe_float(row, "Open"), _safe_float(row, "High"),
                     _safe_float(row, "Low"),  _safe_float(row, "Close"),
                     _safe_float(row, "Volume")))
    if not rows:
        return 0
    last_date = rows[-1][2]
    con = _get_db()
    with _db_lock:
        con.executemany(
            "INSERT OR REPLACE INTO price_cache "
            "(stock,tf,date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?,?)", rows)
        # MIN-06 FIX: total rows in DB, not just this batch
        total = con.execute(
            "SELECT count(*) FROM price_cache WHERE stock=? AND tf=?", (stock, tf)
        ).fetchone()[0]
        con.execute(
            "INSERT OR REPLACE INTO cache_meta "
            "(stock,tf,last_date,last_updated,bar_count) VALUES (?,?,?,?,?)",
            (stock, tf, last_date, str(_today()), total))
        con.commit()
    return len(rows)


def read_cache(stock: str, tf: str = "1mo", limit: int = 9999) -> pd.DataFrame | None:
    """Read OHLCV — uses per-thread connection (MIN-03 FIX)."""
    try:
        con = _get_read_con()
        df = pd.read_sql(
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


def get_last_date(stock: str, tf: str) -> str | None:
    try:
        row = _get_read_con().execute(
            "SELECT last_date FROM cache_meta WHERE stock=? AND tf=?", (stock, tf)
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# HELPER MATH
# ═══════════════════════════════════════════════════════════════════════
def _pct_ratio(val) -> float | None:
    """
    BUG-14 FIX: yfinance growth/margin fields are ALWAYS 0-1 decimals.
    No heuristic. Always multiply by 100.
    Use for: revenueGrowth, earningsGrowth, returnOnEquity, *Margins, etc.
    """
    if val is None:
        return None
    try:
        f = float(val)
        return None if f != f else round(f * 100, 2)
    except Exception:
        return None


def _pct(val) -> float | None:
    """For fields that are already in percentage form (e.g. dividendYield as decimal)."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if f != f else round(f * 100, 2)
    except Exception:
        return None


def _cagr(start, end, years: int) -> float | None:
    """
    SIG-18 FIX: handles turnaround (negative start year).
    - Both negative → None (meaningless)
    - start≤0, end>0 → capped turnaround signal (max 100%)
    - start>0, end≤0 → -100%
    - Normal → standard CAGR formula
    """
    if years <= 0 or start is None or end is None:
        return None
    try:
        s, e = float(start), float(end)
    except Exception:
        return None
    if s < 0 and e < 0:
        return None
    if s <= 0 and e > 0:
        return min(100.0, round(e / max(abs(s), 1) * 30, 1))
    if s > 0 and e <= 0:
        return -100.0
    if s == 0:
        return None
    try:
        return round(((e / s) ** (1 / years) - 1) * 100, 2)
    except Exception:
        return None


def _growth(prev, curr) -> float | None:
    if prev is None or prev == 0:
        return None
    try:
        return round((float(curr) - float(prev)) / abs(float(prev)) * 100, 2)
    except Exception:
        return None


def _trend_slope(series: list) -> float | None:
    """Linear slope over a list. Positive = improving, negative = declining."""
    clean = [float(v) for v in (series or []) if v is not None]
    if len(clean) < 2:
        return None
    x = np.arange(len(clean), dtype=float)
    try:
        return round(float(np.polyfit(x, clean, 1)[0]), 4)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# FUNDAMENTAL FETCH — yfinance
# ═══════════════════════════════════════════════════════════════════════
def fetch_yf_fundamentals(sym: str) -> dict:
    result = {"_yf_ok": False}
    for attempt in range(DL_RETRIES):
        try:
            sess = _get_session()
            kw   = {"session": sess} if sess else {}
            tk   = yf.Ticker(sym, **kw)
            info = tk.info or {}
            if not info.get("marketCap") and not info.get("regularMarketPrice"):
                if attempt < DL_RETRIES - 1:
                    time.sleep(4); continue
                return result

            # Analyst / earnings
            analyst_count = info.get("numberOfAnalystOpinions") or 0
            next_earnings = None
            try:
                cal = tk.calendar
                if cal is not None:
                    ne = cal.get("Earnings Date")
                    if ne and len(ne) > 0:
                        next_earnings = str(ne[0])[:10]
            except Exception:
                pass

            # GAP-07: Dividend growth from history
            dividend_growth_3yr = None
            try:
                div_hist = tk.dividends
                if div_hist is not None and len(div_hist) >= 2:
                    annual = div_hist.resample("YE").sum()
                    if len(annual) >= 4:
                        dividend_growth_3yr = _cagr(float(annual.iloc[-4]), float(annual.iloc[-1]), 3)
                    elif len(annual) >= 2:
                        dividend_growth_3yr = _growth(float(annual.iloc[-2]), float(annual.iloc[-1]))
            except Exception:
                pass

            result.update({
                "_yf_ok":                    True,
                "long_name":                 info.get("longName") or info.get("shortName"),
                "sector":                    info.get("sector"),
                "industry":                  info.get("industry"),
                "market_cap":                info.get("marketCap"),
                "shares_outstanding":        info.get("sharesOutstanding"),
                "float_shares":              info.get("floatShares"),
                # Valuation
                "pe_ratio":                  info.get("trailingPE"),
                "forward_pe":                info.get("forwardPE"),
                "pb_ratio":                  info.get("priceToBook"),
                "ev_ebitda":                 info.get("enterpriseToEbitda"),
                "ev_revenue":                info.get("enterpriseToRevenue"),
                "dividend_yield":            _pct(info.get("dividendYield")),
                "payout_ratio":              _pct_ratio(info.get("payoutRatio")),
                "dividend_growth_3yr":       dividend_growth_3yr,
                # BUG-14 FIX: always _pct_ratio() for yf growth/margin fields
                "roe":                       _pct_ratio(info.get("returnOnEquity")),
                "roa":                       _pct_ratio(info.get("returnOnAssets")),
                "operating_margin":          _pct_ratio(info.get("operatingMargins")),
                "gross_margin":              _pct_ratio(info.get("grossMargins")),
                "net_margin":                _pct_ratio(info.get("profitMargins")),
                "revenue_growth_ttm":        _pct_ratio(info.get("revenueGrowth")),
                "earnings_growth_ttm":       _pct_ratio(info.get("earningsGrowth")),
                "earnings_quarterly_growth": _pct_ratio(info.get("earningsQuarterlyGrowth")),
                # Balance sheet
                "debt_to_equity":            info.get("debtToEquity"),   # raw % from yf
                "current_ratio":             info.get("currentRatio"),
                "quick_ratio":               info.get("quickRatio"),
                "total_debt":                info.get("totalDebt"),
                "total_cash":                info.get("totalCash"),
                "total_revenue":             info.get("totalRevenue"),
                "ebitda":                    info.get("ebitda"),
                "free_cashflow":             info.get("freeCashflow"),
                "operating_cashflow":        info.get("operatingCashflow"),
                # Holdings
                "held_pct_institutions":     _pct_ratio(info.get("heldPercentInstitutions")),
                "held_pct_insiders":         _pct_ratio(info.get("heldPercentInsiders")),
                # Price
                "fifty_two_week_high":       info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low":        info.get("fiftyTwoWeekLow"),
                "fifty_day_avg":             info.get("fiftyDayAverage"),
                "two_hundred_day_avg":       info.get("twoHundredDayAverage"),
                "beta":                      info.get("beta"),
                "regular_market_price":      info.get("regularMarketPrice"),
                # Analyst
                "analyst_target":            info.get("targetMeanPrice"),
                "analyst_high":              info.get("targetHighPrice"),
                "analyst_low":               info.get("targetLowPrice"),
                "analyst_count":             analyst_count,
                "recommendation":            info.get("recommendationKey"),
                "next_earnings":             next_earnings,
            })
            return result
        except Exception as e:
            msg = str(e)
            if "401" in msg or "Crumb" in msg:
                _reset_session(); time.sleep(8)
            elif attempt < DL_RETRIES - 1:
                time.sleep(DL_BACKOFF * (attempt + 1))
    return result


# ═══════════════════════════════════════════════════════════════════════
# SCREENER.IN  (Indian-specific + historical data)
# ═══════════════════════════════════════════════════════════════════════
_SC_SESSION = None
_SC_LOCK    = threading.Lock()


def _get_screener_session():
    global _SC_SESSION
    with _SC_LOCK:
        if _SC_SESSION is None:
            s = requests.Session()
            s.headers.update({
                "User-Agent":     ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                   "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"),
                "Accept":         "text/html,application/xhtml+xml,*/*;q=0.8",
                "Accept-Language":"en-US,en;q=0.5",
                "Referer":        "https://www.screener.in/",
            })
            try:
                s.get("https://www.screener.in/", timeout=15)
            except Exception:
                pass
            _SC_SESSION = s
        return _SC_SESSION


def _parse_screener_num(text: str) -> float | None:
    if not text or text.strip() in ("-", "–", "N/A", ""):
        return None
    cleaned = (text.replace(",", "").replace("₹", "").replace("%", "")
               .replace("Cr.", "").replace("Cr", "").strip())
    if "/" in cleaned:
        cleaned = cleaned.split("/")[0].strip()
    try:
        return round(float(cleaned), 4)
    except Exception:
        return None


def _parse_pl_table(table, result: dict):
    """
    GAP-02/03 FIX: store full profit_history / sales_history / opm_history lists
    so scanner can compute EPS consistency, margin trend, and debt trend.
    """
    rows_map = {}
    for tr in table.find_all("tr"):
        tds = tr.find_all(["td", "th"])
        if not tds:
            continue
        label = tds[0].get_text(strip=True).lower().rstrip("+")
        vals  = [_parse_screener_num(td.get_text(strip=True)) for td in tds[1:]]
        rows_map[label] = vals

    # Sales
    sales = rows_map.get("sales") or rows_map.get("revenue")
    if sales:
        result["sales_history"]    = sales[-6:]        # raw (may have None)
        clean = [v for v in sales if v and v > 0]
        if len(clean) >= 4:
            result["sales_3yr_cagr"]  = _cagr(clean[-4], clean[-1], 3)
        if len(clean) >= 6:
            result["sales_5yr_cagr"]  = _cagr(clean[-6], clean[-1], 5)
        if len(clean) >= 2:
            result["sales_growth_ttm"] = _growth(clean[-2], clean[-1])

    # Net Profit (keep negatives in history for SIG-18 / EPS consistency)
    profit = rows_map.get("net profit") or rows_map.get("profit after tax")
    if profit:
        result["profit_history"] = profit[-6:]         # raw (includes negatives)
        pos = [v for v in profit if v and v > 0]
        if len(pos) >= 4:
            result["profit_3yr_cagr"] = _cagr(pos[-4], pos[-1], 3)
        if len(pos) >= 6:
            result["profit_5yr_cagr"] = _cagr(pos[-6], pos[-1], 5)
        all_vals = [(v or 0.0) for v in profit]
        if len(all_vals) >= 2 and all_vals[-2] != 0:
            result["profit_growth_ttm"] = _growth(all_vals[-2], all_vals[-1])

    # OPM % (SIG-07 / GAP-04 trend)
    opm = rows_map.get("opm %") or rows_map.get("operating profit margin %")
    if opm:
        clean_opm = [v for v in opm if v is not None]
        result["opm_history"] = clean_opm[-6:]
        if clean_opm:
            result["operating_margin_screener"] = clean_opm[-1]


def _parse_shareholding(section, result: dict):
    """
    GAP-01 / GAP-08 FIX: store last 4 quarters of FII/DII/promoter history.
    """
    try:
        table = section.find("table")
        if not table:
            return
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            label = tds[0].get_text(strip=True).lower()
            vals  = [_parse_screener_num(td.get_text(strip=True)) for td in tds[1:]]
            vals  = [v for v in vals if v is not None]
            if not vals:
                continue
            hist4 = vals[-4:]

            if "promoter" in label and "pledge" not in label:
                result["promoter_holding"] = vals[-1]
                result["promoter_history"] = hist4    # GAP-08
            elif "pledge" in label:
                result["promoter_pledge"]  = vals[-1]
            elif "fii" in label or "foreign" in label:
                result["fii_holding"]      = vals[-1]
                result["fii_history"]      = hist4    # GAP-01
            elif "dii" in label or "domestic inst" in label:
                result["dii_holding"]      = vals[-1]
                result["dii_history"]      = hist4    # GAP-01
            elif "public" in label:
                result["public_holding"]   = vals[-1]
    except Exception as e:
        log.debug(f"Shareholding parse: {e}")


def _parse_ratios_history(ratio_sec, result: dict):
    """
    GAP-02 / GAP-04 FIX: store historical D/E and ROCE as lists.
    Scanner computes trend slope from these.
    """
    try:
        table = ratio_sec.find("table")
        if not table:
            return
        tbody = table.find("tbody") or table
        for tr in tbody.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            label  = tds[0].get_text(strip=True).lower()
            vals   = [_parse_screener_num(td.get_text(strip=True)) for td in tds[1:]]
            clean  = [v for v in vals if v is not None]
            latest = clean[-1] if clean else None

            if "roce" in label:
                result.setdefault("roce", latest)
                result["roce_history"] = clean[-6:]
            elif "roe" in label:
                result.setdefault("roe_screener", latest)
            elif "debt / equity" in label or "debt/equity" in label:
                result["debt_equity"] = latest
                result["de_history"]  = clean[-6:]    # GAP-02
            elif "interest coverage" in label:
                result["interest_coverage"] = latest
            elif "asset turnover" in label:
                result["asset_turnover"] = latest
            elif "current ratio" in label:
                result.setdefault("current_ratio_screener", latest)
    except Exception as e:
        log.debug(f"Ratios history: {e}")


def fetch_screener(symbol: str) -> dict:
    if not BS4_OK:
        return {"_screener_ok": False, "_skip": "bs4 not installed"}

    sym    = symbol.replace(".NS", "").replace(".BO", "").upper()
    result = {"_screener_ok": False}
    sess   = _get_screener_session()

    for suffix in ["/consolidated/", "/"]:
        url = f"https://www.screener.in/company/{sym}{suffix}"
        try:
            resp = sess.get(url, timeout=20)
            if resp.status_code == 404:
                continue
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "lxml")

            # Top ratios
            top = soup.find("ul", id="top-ratios") or soup.find("div", id="top-ratios")
            if top:
                for li in top.find_all("li"):
                    spans = li.find_all("span")
                    if len(spans) < 2:
                        continue
                    key = spans[0].get_text(strip=True).lower()
                    val = _parse_screener_num(spans[-1].get_text(strip=True))
                    if "roce" in key:         result["roce"] = val
                    elif "roe" in key:        result["roe_screener"] = val
                    elif "p/e" in key:        result["pe_screener"] = val
                    elif "book value" in key: result["book_value_screener"] = val
                    elif "dividend yield" in key: result["dividend_yield_screener"] = val
                    elif "market cap" in key: result["market_cap_cr_screener"] = val
                    elif "face value" in key: result["face_value"] = val

            # P&L
            pl_sec = soup.find("section", id="profit-loss")
            if pl_sec:
                t = pl_sec.find("table")
                if t:
                    _parse_pl_table(t, result)

            # Key Ratios with history
            ratio_sec = soup.find("section", id="ratios")
            if ratio_sec:
                _parse_ratios_history(ratio_sec, result)

            # Shareholding with history
            sh_sec = (soup.find("section", id="shareholding")
                      or soup.find("div",     id="shareholding"))
            if sh_sec:
                _parse_shareholding(sh_sec, result)

            result["_screener_ok"]  = True
            result["_screener_url"] = url
            return result

        except Exception as e:
            log.debug(f"Screener {sym}: {e}")
            continue

    return result


# ═══════════════════════════════════════════════════════════════════════
# MERGED FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════════════
def fetch_and_merge_fundamentals(sym: str) -> dict:
    yf_data = fetch_yf_fundamentals(sym)
    time.sleep(0.5)
    sc_data = fetch_screener(sym)

    merged = {**yf_data}
    merged["_sources"] = []
    if yf_data.get("_yf_ok"):
        merged["_sources"].append("yfinance")

    if sc_data.get("_screener_ok"):
        merged["_sources"].append("screener.in")
        for k in ["roce", "roe_screener", "pe_screener", "book_value_screener",
                  "face_value", "market_cap_cr_screener", "debt_equity",
                  "interest_coverage", "promoter_holding", "promoter_pledge",
                  "fii_holding", "dii_holding", "public_holding",
                  "sales_3yr_cagr", "sales_5yr_cagr", "profit_3yr_cagr", "profit_5yr_cagr",
                  "sales_growth_ttm", "profit_growth_ttm",
                  "operating_margin_screener", "dividend_yield_screener",
                  "sales_history", "profit_history", "opm_history",
                  "roce_history", "de_history",
                  "promoter_history", "fii_history", "dii_history",
                  "asset_turnover", "current_ratio_screener"]:
            if k in sc_data and sc_data[k] is not None:
                merged[k] = sc_data[k]

        # GAP-09 FIX: market cap cross-validation
        sc_cap = sc_data.get("market_cap_cr_screener")
        yf_cap = (yf_data.get("market_cap") or 0) / 1e7
        if sc_cap and yf_cap and yf_cap > 0:
            if abs(sc_cap - yf_cap) / yf_cap > 0.50:
                log.warning(f"{sym}: Screener ₹{sc_cap:.0f}Cr vs yfinance ₹{yf_cap:.0f}Cr — possible stale scrape")
                merged["_screener_stale_warning"] = True

    # BUG-10 FIX: yfinance debtToEquity is ALWAYS in % (e.g. 45.0 = 0.45 ratio)
    if merged.get("debt_to_equity") is not None:
        merged["debt_to_equity_ratio"] = round(float(merged["debt_to_equity"]) / 100, 3)
    # Screener debt_equity is already a ratio — overrides yfinance if available
    if merged.get("debt_equity") is not None:
        merged["debt_to_equity_ratio"] = round(float(merged["debt_equity"]), 3)

    # Derived trend metrics (used by scanner for GAP-01/02/04/07)
    merged["opm_trend"]       = _trend_slope(merged.get("opm_history",  []))
    merged["roce_trend"]      = _trend_slope(merged.get("roce_history", []))
    merged["de_trend"]        = _trend_slope(merged.get("de_history",   []))  # negative = improving

    fii_h = merged.get("fii_history", [])
    dii_h = merged.get("dii_history", [])
    pro_h = merged.get("promoter_history", [])
    merged["fii_delta"]      = round(fii_h[-1] - fii_h[-2], 2) if len(fii_h) >= 2 else None
    merged["dii_delta"]      = round(dii_h[-1] - dii_h[-2], 2) if len(dii_h) >= 2 else None
    merged["promoter_delta"] = round(pro_h[-1] - pro_h[-2], 2) if len(pro_h) >= 2 else None

    merged["_updated"] = str(_today())
    return merged


# ═══════════════════════════════════════════════════════════════════════
# FUNDAMENTALS CACHE
# ═══════════════════════════════════════════════════════════════════════
def write_fund(stock: str, fund: dict):
    con = _get_db()
    with _db_lock:
        con.execute(
            "INSERT OR REPLACE INTO fund_cache (stock,fund_json,updated_date) VALUES (?,?,?)",
            (stock, json.dumps(fund), str(_today())))
        con.commit()


def read_fund(stock: str) -> dict | None:
    try:
        row = _get_read_con().execute(
            "SELECT fund_json, updated_date FROM fund_cache WHERE stock=?", (stock,)
        ).fetchone()
        if not row or not row[0]:
            return None
        updated = datetime.strptime(row[1], "%Y-%m-%d").date()
        if (_today() - updated).days > FUND_REFRESH_DAYS:
            return None
        return json.loads(row[0])
    except Exception:
        return None


def get_fund_cached_or_fetch(sym: str) -> dict:
    cached = read_fund(sym)
    if cached is not None:
        return cached
    fund = fetch_and_merge_fundamentals(sym)
    if fund.get("_yf_ok") or fund.get("_screener_ok"):
        write_fund(sym, fund)
        time.sleep(SCREENER_DELAY)
    return fund


# ═══════════════════════════════════════════════════════════════════════
# UNIVERSE
# ═══════════════════════════════════════════════════════════════════════
def load_universe() -> list[str]:
    """
    SIG-14 FIX: strict SERIES column guard.
    If column absent → hard-fail to cache fallback (never silently return all 5000+).
    """
    urls = [
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    ]
    for url in urls:
        for attempt in range(2):
            try:
                resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                df = pd.read_csv(StringIO(resp.text)).dropna(subset=["SYMBOL"])

                # SIG-14 FIX: strict column check
                series_col = next((c for c in [" SERIES", "SERIES", " Series", "Series"]
                                   if c in df.columns), None)
                if series_col is None:
                    log.error("NSE CSV missing SERIES column — cannot safely filter. "
                              "Falling back to cache.")
                    break

                df   = df[df[series_col].astype(str).str.strip() == "EQ"]
                syms = [s.strip() + ".NS" for s in df["SYMBOL"].astype(str).tolist()]
                log.info(f"Universe: {len(syms)} EQ stocks from NSE ({series_col} filtered)")
                return syms
            except Exception as e:
                log.warning(f"Universe {url} attempt {attempt+1}: {e}")
                time.sleep(3)

    log.warning("NSE URL blocked — using cached stock list")
    rows = _get_read_con().execute(
        "SELECT DISTINCT stock FROM cache_meta WHERE tf='1mo'"
    ).fetchall()
    return [r[0] for r in rows] if rows else []


# ═══════════════════════════════════════════════════════════════════════
# INDEX / SECTOR
# ═══════════════════════════════════════════════════════════════════════
def update_indices():
    log.info("Updating index data...")
    con = _get_db()
    for name, sym in INDICES.items():
        for tf in ["1mo", "1wk"]:
            try:
                last = get_last_date(sym, tf)
                df   = dl_since(sym, tf, last[:10]) if last else dl_ohlcv(sym, tf, TF_CONFIG.get(tf, "10y"))
                if df is None or len(df) == 0:
                    continue
                rows = []
                for idx, row in df.iterrows():
                    d = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)[:10]
                    rows.append((sym, tf, d,
                                 _safe_float(row, "Open"), _safe_float(row, "High"),
                                 _safe_float(row, "Low"),  _safe_float(row, "Close"),
                                 _safe_float(row, "Volume")))
                with _db_lock:
                    con.executemany("INSERT OR REPLACE INTO index_cache "
                                    "(symbol,tf,date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?,?)", rows)
                    con.executemany("INSERT OR REPLACE INTO price_cache "
                                    "(stock,tf,date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?,?)",
                                    [(sym, tf, r[2], r[3], r[4], r[5], r[6], r[7]) for r in rows])
                    total = con.execute("SELECT count(*) FROM price_cache WHERE stock=? AND tf=?",
                                        (sym, tf)).fetchone()[0]
                    con.execute("INSERT OR REPLACE INTO cache_meta "
                                "(stock,tf,last_date,last_updated,bar_count) VALUES (?,?,?,?,?)",
                                (sym, tf, rows[-1][2], str(_today()), total))
                    con.commit()
                log.info(f"  {name} {tf}: {len(rows)} bars")
                time.sleep(0.5)
            except Exception as e:
                log.warning(f"Index {name} {tf}: {e}")


# ═══════════════════════════════════════════════════════════════════════
# BATCH RUNNERS
# ═══════════════════════════════════════════════════════════════════════
def update_stock_ohlcv(sym: str) -> dict:
    r = {"sym": sym, "ok": 0, "skip": 0, "err": 0}
    for tf, period in TF_CONFIG.items():
        try:
            last = get_last_date(sym, tf)
            df   = dl_since(sym, tf, last[:10]) if last else dl_ohlcv(sym, tf, period)
            if df is None or len(df) == 0:
                r["skip"] += 1; continue
            write_cache(sym, tf, df)
            r["ok"] += 1
        except Exception as e:
            log.debug(f"{sym} {tf}: {e}"); r["err"] += 1
    return r


def run_ohlcv_update(stocks: list[str]):
    log.info(f"OHLCV update: {len(stocks)} stocks")
    t0 = time.time(); ok = err = skip = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(update_stock_ohlcv, s): s for s in stocks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                r = fut.result(); ok += r["ok"]; skip += r["skip"]; err += r["err"]
            except Exception:
                err += 1
            if done % 100 == 0:
                eta = (len(stocks)-done) / max(done/(time.time()-t0), 0.001)
                log.info(f"  OHLCV {done}/{len(stocks)} ok={ok} skip={skip} err={err} ETA={eta:.0f}s")
    log.info(f"OHLCV done: {time.time()-t0:.0f}s")


def run_fundamentals_update(stocks: list[str], workers: int = 2, force: bool = False):
    """
    BUG-11 / SIG-09 FIX: FUND_REFRESH_DAYS=30 means bi-weekly runs
    won't re-download everything. `force=True` clears cache first.
    """
    if force:
        con = _get_db()
        with _db_lock:
            con.execute("DELETE FROM fund_cache"); con.commit()
        log.info("Fundamentals cache cleared")

    stale = [s for s in stocks if read_fund(s) is None]
    log.info(f"Fundamentals: {len(stale)}/{len(stocks)} stale (>{FUND_REFRESH_DAYS}d)")
    if not stale:
        log.info("All fundamentals fresh — skipping"); return

    t0 = time.time(); ok = err = 0

    def _update_one(sym):
        fund = fetch_and_merge_fundamentals(sym)
        if fund.get("_yf_ok") or fund.get("_screener_ok"):
            write_fund(sym, fund); time.sleep(SCREENER_DELAY); return "ok"
        return "err"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_update_one, s): s for s in stale}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                if fut.result() == "ok": ok += 1
                else: err += 1
            except Exception:
                err += 1
            if done % 50 == 0:
                log.info(f"  Fundamentals {done}/{len(stale)} ok={ok} err={err}")
    log.info(f"Fundamentals done: {time.time()-t0:.0f}s ok={ok} err={err}")


def run_bootstrap(stocks: list[str]):
    log.info(f"=== BOOTSTRAP: {len(stocks)} stocks (est. 90-120 min) ===")
    log.info("Phase 1/3: Index data")
    update_indices()
    log.info("Phase 2/3: OHLCV")
    run_ohlcv_update(stocks)
    log.info("Phase 3/3: Fundamentals")
    run_fundamentals_update(stocks, workers=2)
    log.info("Bootstrap complete.")


def run_update(stocks: list[str]):
    log.info(f"=== INCREMENTAL UPDATE: {len(stocks)} stocks ===")
    update_indices()
    run_ohlcv_update(stocks)
    run_fundamentals_update(stocks, workers=2, force=False)


def print_stats():
    con = _get_db()
    print("\n=== OHLCV Cache ===")
    rows = con.execute("""SELECT tf, count(distinct stock), sum(bar_count),
                          min(last_date), max(last_date)
                          FROM cache_meta GROUP BY tf ORDER BY tf""").fetchall()
    print(f"{'TF':<6} {'Stocks':>8} {'Total Bars':>12} {'Oldest':<14} {'Newest':<14}")
    print("-"*58)
    for r in rows:
        print(f"{r[0]:<6} {r[1]:>8,} {(r[2] or 0):>12,} {str(r[3]):<14} {str(r[4]):<14}")
    n_fund  = con.execute("SELECT count(*) FROM fund_cache").fetchone()[0]
    n_fresh = con.execute(
        "SELECT count(*) FROM fund_cache WHERE julianday('now')-julianday(updated_date)<=?",
        (FUND_REFRESH_DAYS,)).fetchone()[0]
    n_idx = con.execute("SELECT count(distinct symbol) FROM index_cache").fetchone()[0]
    db_mb = os.path.getsize(CACHE_PATH) / 1e6
    print(f"\nFundamentals: {n_fund} cached ({n_fresh} fresh / {FUND_REFRESH_DAYS}d window)")
    print(f"Indices: {n_idx} | DB: {db_mb:.1f} MB → {CACHE_PATH}")


def main():
    ap = argparse.ArgumentParser(description="NSE Long-Term Investment Data Updater v2")
    m  = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--bootstrap",    action="store_true")
    m.add_argument("--update",       action="store_true")
    m.add_argument("--fundamentals", action="store_true", help="Force-refresh all fundamentals")
    m.add_argument("--sectors",      action="store_true")
    m.add_argument("--stats",        action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    _get_db()

    if args.stats:
        print_stats(); return
    if args.sectors:
        update_indices(); return

    stocks = load_universe()
    if args.limit > 0:
        stocks = stocks[:args.limit]
        log.info(f"Limited to {len(stocks)}")

    if args.bootstrap:     run_bootstrap(stocks)
    elif args.update:      run_update(stocks)
    elif args.fundamentals: run_fundamentals_update(stocks, workers=2, force=True)


if __name__ == "__main__":
    main()
