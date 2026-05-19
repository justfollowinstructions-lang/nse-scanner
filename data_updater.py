#!/usr/bin/env python3
"""
NSE Long-Term Investment Data Updater
======================================
Fetches and caches multi-timeframe OHLCV + fundamentals from multiple sources.

Sources:
  1. yfinance     — Monthly / Weekly / Daily OHLCV + basic fundamentals
  2. Screener.in  — Indian-specific fundamentals (ROE, ROCE, D/E, promoter)
  3. NSE India    — Stock universe

Timeframes stored:
  1mo  → max history (~10-20 yr)  — primary for pattern detection
  1wk  → 10 yr                    — confirmation
  1d   → 2 yr                     — entry / context

Modes:
  --bootstrap    Full first-time download (run once, ~90-120 min for full universe)
  --update       Incremental OHLCV update (run bi-weekly)
  --fundamentals Refresh fundamental data only (run weekly)
  --sectors      Update index / sector data
  --stats        Print DB statistics

Usage:
  python data_updater.py --bootstrap
  python data_updater.py --update
  python data_updater.py --fundamentals
  python data_updater.py --sectors
  python data_updater.py --stats
"""

import os, sys, json, time, sqlite3, argparse, logging, re
from datetime import date, datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

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
            os.path.join(LOG_DIR, f"updater_{date.today()}.log"),
            encoding="utf-8"),
    ],
)
log = logging.getLogger("data_updater")

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
_IST = timezone(timedelta(hours=5, minutes=30))
def _now():   return datetime.now(_IST)
def _today(): return _now().date()

MAX_WORKERS           = 4     # Conservative — avoids Yahoo 429s on CI
DL_RETRIES            = 3
DL_BACKOFF            = 4.0
FUND_REFRESH_DAYS     = 7     # Re-fetch fundamentals if older than this
SCREENER_DELAY        = 2.5   # Seconds between screener.in requests (be polite)
MIN_MARKET_CAP_CR     = 200   # Skip nano-caps < ₹200 Cr

# Timeframes and their fetch periods
TF_CONFIG = {
    "1mo": "max",   # All available monthly data
    "1wk": "10y",   # 10 years of weekly
    "1d":  "2y",    # 2 years of daily
}

# Major indices for market context (fetched as sector data)
INDICES = {
    "NIFTY50":        "^NSEI",
    "NIFTYBANK":      "^NSEBANK",
    "NIFTYMIDCAP50":  "^NSEMDCP50",
    "INDIAVIX":       "^INDIAVIX",
    "NIFTYIT":        "^CNXIT",
    "NIFTYPHARMA":    "^CNXPHARMA",
    "NIFTYAUTO":      "^CNXAUTO",
    "NIFTYFMCG":      "^CNXFMCG",
    "NIFTYMETAL":     "^CNXMETAL",
    "NIFTYINFRA":     "^CNXINFRA",
    "NIFTYENERGY":    "^CNXENERGY",
    "NIFTYREALTY":    "^CNXREALTY",
    "NIFTYCONSUMPTION":"^CNXCONSUM",
    "NIFTYPSUBANK":   "^CNXPSUBANK",
}

# ═══════════════════════════════════════════════════════════════════════
# DATABASE SETUP
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
            _db_con.execute("PRAGMA cache_size=-131072")   # 128 MB page cache
            _db_con.executescript("""
                -- Multi-timeframe OHLCV cache (monthly, weekly, daily)
                CREATE TABLE IF NOT EXISTS price_cache (
                    stock  TEXT    NOT NULL,
                    tf     TEXT    NOT NULL,
                    date   TEXT    NOT NULL,
                    open   REAL,
                    high   REAL,
                    low    REAL,
                    close  REAL    NOT NULL,
                    volume REAL,
                    PRIMARY KEY (stock, tf, date)
                );

                -- Last-updated metadata
                CREATE TABLE IF NOT EXISTS cache_meta (
                    stock        TEXT NOT NULL,
                    tf           TEXT NOT NULL,
                    last_date    TEXT,
                    last_updated TEXT,
                    bar_count    INTEGER,
                    PRIMARY KEY (stock, tf)
                );

                -- Fundamental data from yfinance + screener.in
                CREATE TABLE IF NOT EXISTS fund_cache (
                    stock        TEXT PRIMARY KEY,
                    fund_json    TEXT,
                    updated_date TEXT
                );

                -- Index / sector OHLCV (monthly + weekly)
                CREATE TABLE IF NOT EXISTS index_cache (
                    symbol TEXT NOT NULL,
                    tf     TEXT NOT NULL,
                    date   TEXT NOT NULL,
                    open   REAL,
                    high   REAL,
                    low    REAL,
                    close  REAL NOT NULL,
                    volume REAL,
                    PRIMARY KEY (symbol, tf, date)
                );

                -- Investment scan results history
                CREATE TABLE IF NOT EXISTS scan_results (
                    scan_id    TEXT NOT NULL,
                    scan_date  TEXT NOT NULL,
                    stock      TEXT NOT NULL,
                    score      REAL,
                    grade      TEXT,
                    result_json TEXT,
                    PRIMARY KEY (scan_id, stock)
                );

                -- Per-stock history across scans
                CREATE TABLE IF NOT EXISTS stock_history (
                    stock             TEXT PRIMARY KEY,
                    first_seen_date   TEXT,
                    first_price       REAL,
                    first_score       REAL,
                    last_seen_date    TEXT,
                    last_price        REAL,
                    last_score        REAL,
                    times_scanned     INTEGER DEFAULT 0,
                    pattern_history   TEXT    -- JSON list
                );

                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_pc_stock_tf ON price_cache(stock, tf);
                CREATE INDEX IF NOT EXISTS idx_pc_date     ON price_cache(tf, date);
                CREATE INDEX IF NOT EXISTS idx_ic_sym_tf   ON index_cache(symbol, tf);
                CREATE INDEX IF NOT EXISTS idx_meta        ON cache_meta(stock, tf);
            """)
            _db_con.commit()
        return _db_con


# ═══════════════════════════════════════════════════════════════════════
# YAHOO FINANCE SESSION  (Chrome impersonation avoids 401 on CI)
# ═══════════════════════════════════════════════════════════════════════
_YF_SESSION      = None
_YF_SESSION_LOCK = Lock()

def _build_session():
    try:
        from curl_cffi import requests as _cr
        sess = _cr.Session(impersonate="chrome110")
        sess.get("https://finance.yahoo.com", timeout=15)
        log.info("curl_cffi Chrome session ready")
        return sess
    except ImportError:
        log.warning("curl_cffi not installed — may get 401. pip install curl_cffi")
        return None
    except Exception as e:
        log.warning(f"Session build failed: {e}")
        return None

def _get_session():
    global _YF_SESSION
    with _YF_SESSION_LOCK:
        if _YF_SESSION is None:
            _YF_SESSION = _build_session()
        return _YF_SESSION

def _reset_session():
    global _YF_SESSION
    with _YF_SESSION_LOCK:
        _YF_SESSION = None


# ═══════════════════════════════════════════════════════════════════════
# OHLCV DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════
def dl_ohlcv(sym: str, tf: str = "1mo", period: str = "max") -> pd.DataFrame | None:
    """Download OHLCV bars. Returns clean DataFrame or None."""
    for attempt in range(DL_RETRIES):
        try:
            sess = _get_session()
            kw   = {"session": sess} if sess else {}
            import contextlib, io as _io
            with contextlib.redirect_stderr(_io.StringIO()):
                df = yf.download(sym, period=period, interval=tf,
                                 auto_adjust=True, progress=False, timeout=30, **kw)
            if df is None or len(df) == 0:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close"])
            df.columns = [c.capitalize() if c.lower() in
                          ["open","high","low","close","volume"] else c
                          for c in df.columns]
            return df if len(df) >= 2 else None
        except Exception as e:
            msg = str(e)
            if "401" in msg or "Crumb" in msg or "Unauthorized" in msg:
                log.warning(f"401 {sym} attempt {attempt+1} — resetting session")
                _reset_session(); time.sleep(10)
            elif "429" in msg or "rate limit" in msg.lower():
                wait = 20 * (attempt + 1)
                log.warning(f"RateLimit {sym} — waiting {wait}s")
                time.sleep(wait)
            elif "delisted" in msg.lower() or "no price data" in msg.lower():
                return None
            elif attempt < DL_RETRIES - 1:
                time.sleep(DL_BACKOFF * (attempt + 1))
    return None


def dl_since(sym: str, tf: str, since: str) -> pd.DataFrame | None:
    """Download only bars after a given date."""
    for attempt in range(DL_RETRIES):
        try:
            sess = _get_session()
            kw   = {"session": sess} if sess else {}
            import contextlib, io as _io
            with contextlib.redirect_stderr(_io.StringIO()):
                df = yf.download(sym, start=since, interval=tf,
                                 auto_adjust=True, progress=False, timeout=30, **kw)
            if df is None or len(df) == 0:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close"])
            df.columns = [c.capitalize() if c.lower() in
                          ["open","high","low","close","volume"] else c
                          for c in df.columns]
            return df if len(df) >= 1 else None
        except Exception as e:
            msg = str(e)
            if "401" in msg or "Crumb" in msg:
                _reset_session(); time.sleep(8)
            elif "delisted" in msg.lower():
                return None
            elif attempt < DL_RETRIES - 1:
                time.sleep(DL_BACKOFF * (attempt + 1))
    return None


# ═══════════════════════════════════════════════════════════════════════
# CACHE READ / WRITE
# ═══════════════════════════════════════════════════════════════════════
def write_cache(stock: str, tf: str, df: pd.DataFrame) -> int:
    """Upsert OHLCV rows. Returns rows written."""
    if df is None or len(df) == 0:
        return 0
    rows = []
    for idx, row in df.iterrows():
        date_str = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)[:10]
        rows.append((
            stock, tf, date_str,
            _safe_float(row, "Open"),  _safe_float(row, "High"),
            _safe_float(row, "Low"),   _safe_float(row, "Close"),
            _safe_float(row, "Volume"),
        ))
    if not rows:
        return 0
    con = _get_db()
    last_date = rows[-1][2]
    with _db_lock:
        con.executemany(
            "INSERT OR REPLACE INTO price_cache "
            "(stock,tf,date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?,?)",
            rows
        )
        con.execute(
            "INSERT OR REPLACE INTO cache_meta (stock,tf,last_date,last_updated,bar_count) "
            "VALUES (?,?,?,?,?)",
            (stock, tf, last_date, str(_today()), len(rows))
        )
        con.commit()
    return len(rows)


def read_cache(stock: str, tf: str = "1mo", limit: int = 9999) -> pd.DataFrame | None:
    """Read cached OHLCV. Returns DataFrame oldest-first or None."""
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


def get_last_date(stock: str, tf: str) -> str | None:
    try:
        con = _get_db()
        row = con.execute(
            "SELECT last_date FROM cache_meta WHERE stock=? AND tf=?",
            (stock, tf)
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def get_bar_count(stock: str, tf: str) -> int:
    try:
        con = _get_db()
        row = con.execute(
            "SELECT bar_count FROM cache_meta WHERE stock=? AND tf=?",
            (stock, tf)
        ).fetchone()
        return row[0] if row else 0
    except Exception:
        return 0


def _safe_float(row, col: str) -> float:
    v = row.get(col, row.get(col.lower(), 0))
    try:
        f = float(v)
        return 0.0 if (f != f) else f   # NaN check
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════
# FUNDAMENTAL FETCH — yfinance
# ═══════════════════════════════════════════════════════════════════════
def fetch_yf_fundamentals(sym: str) -> dict:
    """Fetch fundamentals from yfinance. Returns dict."""
    result = {"_yf_ok": False}
    for attempt in range(DL_RETRIES):
        try:
            sess = _get_session()
            kw   = {"session": sess} if sess else {}
            tk   = yf.Ticker(sym, **kw)
            info = tk.info or {}

            if not info.get("marketCap") and not info.get("regularMarketPrice"):
                if attempt < DL_RETRIES - 1:
                    time.sleep(4)
                    continue
                return result

            # Earnings calendar
            next_earnings = None
            try:
                cal = tk.calendar
                if cal is not None:
                    ne = cal.get("Earnings Date")
                    if ne and len(ne) > 0:
                        next_earnings = str(ne[0])[:10]
            except Exception:
                pass

            # Analyst targets
            analyst_target = info.get("targetMeanPrice")

            result.update({
                "_yf_ok":                   True,
                "long_name":                info.get("longName") or info.get("shortName"),
                "sector":                   info.get("sector"),
                "industry":                 info.get("industry"),
                "market_cap":               info.get("marketCap"),          # INR
                "shares_outstanding":       info.get("sharesOutstanding"),
                "float_shares":             info.get("floatShares"),
                # Valuation
                "pe_ratio":                 info.get("trailingPE"),
                "forward_pe":               info.get("forwardPE"),
                "pb_ratio":                 info.get("priceToBook"),
                "ev_ebitda":                info.get("enterpriseToEbitda"),
                "ev_revenue":               info.get("enterpriseToRevenue"),
                "dividend_yield":           info.get("dividendYield"),
                "payout_ratio":             info.get("payoutRatio"),
                # Quality
                "roe":                      _pct(info.get("returnOnEquity")),
                "roa":                      _pct(info.get("returnOnAssets")),
                "operating_margin":         _pct(info.get("operatingMargins")),
                "gross_margin":             _pct(info.get("grossMargins")),
                "net_margin":               _pct(info.get("profitMargins")),
                # Growth (TTM YoY)
                "revenue_growth_ttm":       _pct(info.get("revenueGrowth")),
                "earnings_growth_ttm":      _pct(info.get("earningsGrowth")),
                "earnings_quarterly_growth":_pct(info.get("earningsQuarterlyGrowth")),
                # Balance sheet
                "debt_to_equity":           info.get("debtToEquity"),       # in %
                "current_ratio":            info.get("currentRatio"),
                "quick_ratio":              info.get("quickRatio"),
                "total_debt":               info.get("totalDebt"),
                "total_cash":               info.get("totalCash"),
                "total_revenue":            info.get("totalRevenue"),
                "ebitda":                   info.get("ebitda"),
                "free_cashflow":            info.get("freeCashflow"),
                "operating_cashflow":       info.get("operatingCashflow"),
                # Holdings
                "held_pct_institutions":    _pct(info.get("heldPercentInstitutions")),
                "held_pct_insiders":        _pct(info.get("heldPercentInsiders")),
                # Price data
                "fifty_two_week_high":      info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low":       info.get("fiftyTwoWeekLow"),
                "fifty_day_avg":            info.get("fiftyDayAverage"),
                "two_hundred_day_avg":      info.get("twoHundredDayAverage"),
                "beta":                     info.get("beta"),
                "regular_market_price":     info.get("regularMarketPrice"),
                # Analyst
                "analyst_target":           analyst_target,
                "analyst_high":             info.get("targetHighPrice"),
                "analyst_low":              info.get("targetLowPrice"),
                "analyst_count":            info.get("numberOfAnalystOpinions"),
                "recommendation":           info.get("recommendationKey"),
                # Events
                "next_earnings":            next_earnings,
            })
            return result
        except Exception as e:
            msg = str(e)
            if "401" in msg or "Crumb" in msg:
                _reset_session(); time.sleep(8)
            elif attempt < DL_RETRIES - 1:
                time.sleep(DL_BACKOFF * (attempt + 1))
    return result


def _pct(val) -> float | None:
    """Convert 0-1 float to percentage, handle None."""
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:   # NaN
            return None
        # yfinance sometimes returns already-percentage values
        if abs(f) > 5:   # likely already a %, not a ratio
            return round(f, 2)
        return round(f * 100, 2)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# FUNDAMENTAL FETCH — Screener.in  (Indian-specific metrics)
# ═══════════════════════════════════════════════════════════════════════
_SCREENER_SESSION = None
_SCREENER_LOCK    = Lock()

def _get_screener_session():
    global _SCREENER_SESSION
    with _SCREENER_LOCK:
        if _SCREENER_SESSION is None:
            s = requests.Session()
            s.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
                "Accept":           "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language":  "en-US,en;q=0.5",
                "Referer":          "https://www.screener.in/",
            })
            try:
                # Warm up session to get cookies
                s.get("https://www.screener.in/", timeout=15)
            except Exception:
                pass
            _SCREENER_SESSION = s
        return _SCREENER_SESSION


def fetch_screener(symbol: str) -> dict:
    """
    Scrape Screener.in for Indian-specific fundamentals.
    Returns dict with keys: roce, roe_screener, debt_equity, promoter_holding,
    promoter_pledge, sales_growth_3yr, profit_growth_3yr, operating_margin_screener,
    face_value, market_cap_cr_screener, book_value_screener.
    """
    if not BS4_OK:
        return {"_screener_ok": False, "_screener_skip": "bs4 not installed"}

    sym = symbol.replace(".NS", "").replace(".BO", "").upper()
    result = {"_screener_ok": False}

    sess = _get_screener_session()

    for suffix in ["/consolidated/", "/"]:
        url = f"https://www.screener.in/company/{sym}{suffix}"
        try:
            resp = sess.get(url, timeout=20)
            if resp.status_code == 404:
                continue
            if resp.status_code != 200:
                log.debug(f"Screener {sym}: HTTP {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, "lxml")

            # ── Top Ratios (Market cap, PE, Book Val, ROCE, ROE…) ─────────────
            top = soup.find("ul", id="top-ratios")
            if not top:
                top = soup.find("div", id="top-ratios")
            if top:
                for li in top.find_all("li"):
                    spans = li.find_all("span")
                    if len(spans) < 2:
                        continue
                    key = spans[0].get_text(strip=True).lower()
                    raw = spans[-1].get_text(strip=True)
                    val = _parse_screener_num(raw)

                    if "roce" in key:
                        result["roce"] = val
                    elif "roe" in key:
                        result["roe_screener"] = val
                    elif "stock p/e" in key or "p/e" in key:
                        result["pe_screener"] = val
                    elif "book value" in key:
                        result["book_value_screener"] = val
                    elif "dividend yield" in key:
                        result["dividend_yield_screener"] = val
                    elif "market cap" in key:
                        result["market_cap_cr_screener"] = val
                    elif "face value" in key:
                        result["face_value"] = val
                    elif "high / low" in key or "52w" in key.replace(" ", ""):
                        # e.g. "2,345 / 1,123" → keep as string
                        result["high_low_52w"] = raw

            # ── Key Ratios table (D/E, Interest Coverage…) ────────────────────
            ratio_sec = soup.find("section", id="ratios")
            if ratio_sec:
                table = ratio_sec.find("table")
                if table:
                    headers_row = table.find("thead")
                    if headers_row:
                        years = [th.get_text(strip=True)
                                 for th in headers_row.find_all("th")][1:]
                    rows_data = table.find("tbody")
                    if rows_data:
                        for tr in rows_data.find_all("tr"):
                            tds = tr.find_all("td")
                            if not tds:
                                continue
                            label = tds[0].get_text(strip=True).lower()
                            vals  = [_parse_screener_num(td.get_text(strip=True))
                                     for td in tds[1:]]
                            # Use most recent non-None value
                            latest = next((v for v in reversed(vals) if v is not None), None)
                            if latest is None:
                                continue

                            if "debtor days" in label:
                                result["debtor_days"] = latest
                            elif "inventory days" in label:
                                result["inventory_days"] = latest
                            elif "debt / equity" in label or "debt/equity" in label:
                                result["debt_equity"] = latest
                            elif "interest coverage" in label:
                                result["interest_coverage"] = latest
                            elif "roce" in label:
                                result.setdefault("roce", latest)
                            elif "roe" in label:
                                result.setdefault("roe_screener", latest)

            # ── Profit & Loss table (sales/profit growth) ─────────────────────
            pl_sec = soup.find("section", id="profit-loss")
            if pl_sec:
                table = pl_sec.find("table")
                if table:
                    _parse_pl_table(table, result)

            # ── Shareholding pattern ──────────────────────────────────────────
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


def _parse_screener_num(text: str) -> float | None:
    """Parse '1,23,456 Cr.' → 123456.0, '25.3 %' → 25.3, '-' → None."""
    if not text or text.strip() in ("-", "–", "N/A", ""):
        return None
    cleaned = (text.replace(",", "").replace("₹", "").replace("%", "")
               .replace("Cr.", "").replace("Cr", "").strip())
    # Handle ranges like "1234 / 567"
    if "/" in cleaned:
        parts = cleaned.split("/")
        cleaned = parts[0].strip()
    try:
        return round(float(cleaned), 4)
    except Exception:
        return None


def _parse_pl_table(table, result: dict):
    """Parse Profit & Loss table from screener.in to compute 3yr CAGRs."""
    try:
        rows_map = {}
        for tr in table.find_all("tr"):
            tds = tr.find_all(["td", "th"])
            if not tds:
                continue
            label = tds[0].get_text(strip=True).lower().rstrip("+")
            vals  = []
            for td in tds[1:]:
                v = _parse_screener_num(td.get_text(strip=True))
                vals.append(v)
            rows_map[label] = vals

        # Sales row
        sales = rows_map.get("sales") or rows_map.get("revenue")
        if sales:
            clean_sales = [v for v in sales if v and v > 0]
            if len(clean_sales) >= 4:
                result["sales_3yr_cagr"] = _cagr(clean_sales[-4], clean_sales[-1], 3)
            if len(clean_sales) >= 2:
                result["sales_growth_ttm"] = _growth(clean_sales[-2], clean_sales[-1])

        # Net Profit row
        profit = rows_map.get("net profit") or rows_map.get("profit after tax")
        if profit:
            clean_profit = [(v if v else 0) for v in profit]
            non_zero = [v for v in clean_profit if v and v > 0]
            if len(non_zero) >= 4:
                result["profit_3yr_cagr"] = _cagr(non_zero[-4], non_zero[-1], 3)
            if len(clean_profit) >= 2 and clean_profit[-2] and clean_profit[-2] > 0:
                result["profit_growth_ttm"] = _growth(clean_profit[-2], clean_profit[-1])

        # OPM % row
        opm = rows_map.get("opm %") or rows_map.get("operating profit margin %")
        if opm:
            latest_opm = next((v for v in reversed(opm) if v is not None), None)
            if latest_opm is not None:
                result["operating_margin_screener"] = latest_opm

    except Exception as e:
        log.debug(f"P&L parse error: {e}")


def _parse_shareholding(section, result: dict):
    """Parse shareholding section from screener.in HTML."""
    try:
        table = section.find("table")
        if not table:
            return
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            label = tds[0].get_text(strip=True).lower()
            # Most recent value is usually the last column
            vals = []
            for td in tds[1:]:
                v = _parse_screener_num(td.get_text(strip=True))
                if v is not None:
                    vals.append(v)
            if not vals:
                continue
            latest = vals[-1]

            if "promoter" in label and "pledge" not in label:
                result["promoter_holding"] = latest
            elif "pledge" in label:
                result["promoter_pledge"] = latest
            elif "fii" in label or "foreign institutional" in label:
                result["fii_holding"] = latest
            elif "dii" in label or "domestic institutional" in label:
                result["dii_holding"] = latest
            elif "public" in label:
                result["public_holding"] = latest
    except Exception as e:
        log.debug(f"Shareholding parse error: {e}")


def _cagr(start: float, end: float, years: int) -> float | None:
    """Compute CAGR %."""
    if not start or not end or years <= 0 or start <= 0:
        return None
    try:
        return round(((end / start) ** (1 / years) - 1) * 100, 2)
    except Exception:
        return None


def _growth(prev: float, curr: float) -> float | None:
    """YoY growth %."""
    if not prev or prev <= 0:
        return None
    try:
        return round((curr - prev) / abs(prev) * 100, 2)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# MERGED FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════════════
def fetch_and_merge_fundamentals(sym: str) -> dict:
    """
    Fetch from yfinance + Screener.in and merge into a single dict.
    Screener data fills gaps in yfinance (promoter, ROCE, growth CAGRs).
    """
    yf_data  = fetch_yf_fundamentals(sym)
    time.sleep(0.5)   # Small pause before screener
    sc_data  = fetch_screener(sym)

    merged = {**yf_data}   # yfinance is base
    merged["_sources"] = []

    if yf_data.get("_yf_ok"):
        merged["_sources"].append("yfinance")

    if sc_data.get("_screener_ok"):
        merged["_sources"].append("screener.in")
        # Fill in screener-only fields
        for k in ["roce", "roe_screener", "pe_screener", "book_value_screener",
                  "face_value", "market_cap_cr_screener", "debt_equity",
                  "interest_coverage", "promoter_holding", "promoter_pledge",
                  "fii_holding", "dii_holding", "public_holding",
                  "sales_3yr_cagr", "profit_3yr_cagr",
                  "sales_growth_ttm", "profit_growth_ttm",
                  "operating_margin_screener", "debtor_days",
                  "dividend_yield_screener"]:
            if k in sc_data and sc_data[k] is not None:
                merged[k] = sc_data[k]

        # Prefer screener ROCE over yfinance (more accurate for Indian GAAP)
        if "roce" in sc_data and sc_data["roce"] is not None:
            merged["roce"] = sc_data["roce"]

        # D/E: screener uses ratio; yfinance uses % (debtToEquity / 100)
        if "debt_equity" in merged and merged["debt_equity"] is not None:
            merged["debt_to_equity_ratio"] = merged["debt_equity"]
        elif merged.get("debt_to_equity") is not None:
            # yfinance debtToEquity is already a ratio (e.g. 0.45)
            merged["debt_to_equity_ratio"] = round(merged["debt_to_equity"] / 100, 3) \
                if merged["debt_to_equity"] > 5 else merged["debt_to_equity"]

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
            (stock, json.dumps(fund), str(_today()))
        )
        con.commit()


def read_fund(stock: str) -> dict | None:
    """Return cached fundamentals if fresh (within FUND_REFRESH_DAYS)."""
    try:
        con = _get_db()
        row = con.execute(
            "SELECT fund_json, updated_date FROM fund_cache WHERE stock=?",
            (stock,)
        ).fetchone()
        if not row or not row[0]:
            return None
        # Check freshness
        updated = datetime.strptime(row[1], "%Y-%m-%d").date()
        if (_today() - updated).days > FUND_REFRESH_DAYS:
            return None
        return json.loads(row[0])
    except Exception:
        return None


def get_fund_cached_or_fetch(sym: str) -> dict:
    """Return cached fundamentals or fetch fresh."""
    cached = read_fund(sym)
    if cached is not None:
        return cached
    fund = fetch_and_merge_fundamentals(sym)
    if fund.get("_yf_ok") or fund.get("_screener_ok"):
        write_fund(sym, fund)
        time.sleep(SCREENER_DELAY)   # Rate limit Screener.in
    return fund


# ═══════════════════════════════════════════════════════════════════════
# UNIVERSE
# ═══════════════════════════════════════════════════════════════════════
def load_universe(min_cap_cr: float = MIN_MARKET_CAP_CR) -> list[str]:
    """
    Load NSE EQ universe from NSE archives.
    Falls back to cached stock list if URL is blocked.
    Returns list of .NS symbols.
    """
    urls = [
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    ]
    for url in urls:
        for attempt in range(2):
            try:
                resp = requests.get(url, timeout=20,
                                    headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                df = pd.read_csv(StringIO(resp.text)).dropna(subset=["SYMBOL"])
                for col in [" SERIES", "SERIES"]:
                    if col in df.columns:
                        df = df[df[col].str.strip() == "EQ"]
                        break
                syms = [s.strip() + ".NS" for s in df["SYMBOL"].astype(str).tolist()]
                log.info(f"Universe: {len(syms)} EQ stocks from NSE")
                return syms
            except Exception as e:
                log.warning(f"Universe {url} attempt {attempt+1}: {e}")
                time.sleep(3)

    # Fallback: stocks already in cache
    log.warning("NSE URL blocked — using cached stock list")
    con = _get_db()
    rows = con.execute(
        "SELECT DISTINCT stock FROM cache_meta WHERE tf='1mo'"
    ).fetchall()
    return [r[0] for r in rows] if rows else []


# ═══════════════════════════════════════════════════════════════════════
# SECTOR / INDEX UPDATE
# ═══════════════════════════════════════════════════════════════════════
def update_indices():
    """Download monthly + weekly data for all tracked indices."""
    log.info("Updating index data...")
    con = _get_db()
    for name, sym in INDICES.items():
        for tf in ["1mo", "1wk"]:
            try:
                last = get_last_date(sym, tf)
                if last:
                    df = dl_since(sym, tf, last)
                else:
                    period = "max" if tf == "1mo" else "10y"
                    df = dl_ohlcv(sym, tf, period)

                if df is None or len(df) == 0:
                    continue

                rows = []
                for idx, row in df.iterrows():
                    d = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)[:10]
                    rows.append((sym, tf, d,
                                 _safe_float(row, "Open"),  _safe_float(row, "High"),
                                 _safe_float(row, "Low"),   _safe_float(row, "Close"),
                                 _safe_float(row, "Volume")))

                with _db_lock:
                    con.executemany(
                        "INSERT OR REPLACE INTO index_cache "
                        "(symbol,tf,date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?,?)",
                        rows
                    )
                    # Also write to price_cache so scanner can use read_cache
                    con.executemany(
                        "INSERT OR REPLACE INTO price_cache "
                        "(stock,tf,date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?,?)",
                        [(sym, tf, r[2], r[3], r[4], r[5], r[6], r[7]) for r in rows]
                    )
                    con.execute(
                        "INSERT OR REPLACE INTO cache_meta (stock,tf,last_date,last_updated,bar_count)"
                        " VALUES (?,?,?,?,?)",
                        (sym, tf, rows[-1][2], str(_today()), len(rows))
                    )
                    con.commit()

                log.info(f"  {name} ({sym}) {tf}: {len(rows)} bars")
                time.sleep(0.5)
            except Exception as e:
                log.warning(f"Index {name} {tf}: {e}")


def read_index(symbol: str, tf: str = "1mo") -> pd.DataFrame | None:
    """Read index OHLCV from cache."""
    return read_cache(symbol, tf)


# ═══════════════════════════════════════════════════════════════════════
# PER-STOCK OHLCV UPDATE
# ═══════════════════════════════════════════════════════════════════════
def update_stock_ohlcv(sym: str) -> dict:
    """
    Update all three timeframes (1mo, 1wk, 1d) for one stock.
    Downloads only new bars since last stored date.
    """
    result = {"sym": sym, "ok": 0, "skip": 0, "err": 0}

    for tf, period in TF_CONFIG.items():
        try:
            last = get_last_date(sym, tf)
            if last:
                # Incremental: fetch from last date
                df = dl_since(sym, tf, last[:10])
            else:
                # First time: full history
                df = dl_ohlcv(sym, tf, period)

            if df is None or len(df) == 0:
                result["skip"] += 1
                continue

            n = write_cache(sym, tf, df)
            result["ok"] += 1
            log.debug(f"{sym} {tf}: +{n} bars")
        except Exception as e:
            log.debug(f"{sym} {tf} err: {e}")
            result["err"] += 1

    return result


# ═══════════════════════════════════════════════════════════════════════
# BATCH RUNNERS
# ═══════════════════════════════════════════════════════════════════════
def run_ohlcv_update(stocks: list[str]):
    """Parallel OHLCV update for all stocks."""
    log.info(f"OHLCV update: {len(stocks)} stocks × 3 timeframes")
    t0 = time.time()
    ok = err = skip = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(update_stock_ohlcv, s): s for s in stocks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                r = fut.result()
                ok += r["ok"]; skip += r["skip"]; err += r["err"]
            except Exception:
                err += 1
            if done % 100 == 0:
                elapsed = time.time() - t0
                eta = (len(stocks) - done) / (done / elapsed) if done > 0 else 0
                log.info(f"  OHLCV: {done}/{len(stocks)} | ok={ok} skip={skip} err={err} | "
                         f"{elapsed:.0f}s | ETA {eta:.0f}s")

    log.info(f"OHLCV done: {time.time()-t0:.0f}s | ok={ok} skip={skip} err={err}")


def run_fundamentals_update(stocks: list[str], workers: int = 2):
    """
    Update fundamentals for all stocks.
    Uses 2 workers max — Screener.in is rate-limited.
    """
    log.info(f"Fundamentals update: {len(stocks)} stocks")
    t0 = time.time()
    ok = err = skip = 0

    def _update_one(sym):
        existing = read_fund(sym)
        if existing is not None:
            return "skip"
        fund = fetch_and_merge_fundamentals(sym)
        if fund.get("_yf_ok") or fund.get("_screener_ok"):
            write_fund(sym, fund)
            time.sleep(SCREENER_DELAY)
            return "ok"
        return "err"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_update_one, s): s for s in stocks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                status = fut.result()
                if status == "ok":   ok += 1
                elif status == "skip": skip += 1
                else: err += 1
            except Exception:
                err += 1
            if done % 50 == 0:
                log.info(f"  Fundamentals: {done}/{len(stocks)} | ok={ok} skip={skip} err={err}")

    log.info(f"Fundamentals done: {time.time()-t0:.0f}s | ok={ok} skip={skip} err={err}")


def run_bootstrap(stocks: list[str]):
    """
    Full first-time download: OHLCV for all timeframes + fundamentals.
    Estimated time: 60-120 min for 2000+ stocks.
    """
    log.info(f"=== BOOTSTRAP: {len(stocks)} stocks ===")
    log.info("Estimated time: 60-120 minutes. This runs once only.")

    # Phase 1: Indices first (fast, needed for scanner context)
    log.info("Phase 1/3: Index data...")
    update_indices()

    # Phase 2: All OHLCV
    log.info(f"Phase 2/3: OHLCV (monthly + weekly + daily)...")
    run_ohlcv_update(stocks)

    # Phase 3: Fundamentals (slowest — rate limited)
    log.info("Phase 3/3: Fundamentals (yfinance + screener.in)...")
    run_fundamentals_update(stocks, workers=2)

    log.info("Bootstrap complete. Use --update for future incremental runs.")


def run_update(stocks: list[str]):
    """Incremental update: new OHLCV bars + stale fundamentals."""
    log.info(f"=== INCREMENTAL UPDATE: {len(stocks)} stocks ===")

    # Indices
    update_indices()

    # OHLCV (fast — only new bars)
    run_ohlcv_update(stocks)

    # Fundamentals (only refresh stocks where fund data is stale)
    stale = [s for s in stocks if read_fund(s) is None]
    log.info(f"Fundamentals: {len(stale)} stocks need refresh")
    if stale:
        run_fundamentals_update(stale, workers=2)


# ═══════════════════════════════════════════════════════════════════════
# STATS
# ═══════════════════════════════════════════════════════════════════════
def print_stats():
    con = _get_db()
    print("\n=== OHLCV Cache ===")
    rows = con.execute("""
        SELECT tf, count(distinct stock) stocks, count(*) bars,
               min(date) oldest, max(date) newest
        FROM price_cache GROUP BY tf ORDER BY tf
    """).fetchall()
    print(f"{'TF':<6} {'Stocks':>8} {'Bars':>12} {'Oldest':<14} {'Newest':<14}")
    print("-" * 60)
    for r in rows:
        print(f"{r[0]:<6} {r[1]:>8,} {r[2]:>12,} {str(r[3]):<14} {str(r[4]):<14}")

    fund_count = con.execute("SELECT count(*) FROM fund_cache").fetchone()[0]
    fresh = con.execute(
        "SELECT count(*) FROM fund_cache WHERE updated_date >= ?",
        (str(_today() - timedelta(days=FUND_REFRESH_DAYS)),)
    ).fetchone()[0]
    print(f"\nFundamentals: {fund_count} stocks cached ({fresh} fresh)")

    idx_count = con.execute("SELECT count(distinct symbol) FROM index_cache").fetchone()[0]
    print(f"Indices: {idx_count} tracked")

    db_mb = os.path.getsize(CACHE_PATH) / 1e6
    print(f"DB size: {db_mb:.1f} MB → {CACHE_PATH}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="NSE Long-Term Investment Data Updater")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bootstrap",    action="store_true",
                      help="Full first-time download (run once)")
    mode.add_argument("--update",       action="store_true",
                      help="Incremental OHLCV + stale fundamentals")
    mode.add_argument("--fundamentals", action="store_true",
                      help="Refresh all fundamentals (force)")
    mode.add_argument("--sectors",      action="store_true",
                      help="Update index/sector data only")
    mode.add_argument("--stats",        action="store_true",
                      help="Print cache statistics")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit universe size (for testing)")
    args = ap.parse_args()

    # Ensure DB is ready
    _get_db()

    if args.stats:
        print_stats()
        return

    if args.sectors:
        update_indices()
        return

    stocks = load_universe()
    if args.limit > 0:
        stocks = stocks[:args.limit]
        log.info(f"Limited to {len(stocks)} stocks")

    if args.bootstrap:
        run_bootstrap(stocks)
    elif args.update:
        run_update(stocks)
    elif args.fundamentals:
        # Force-refresh all fundamentals
        con = _get_db()
        with _db_lock:
            con.execute("DELETE FROM fund_cache")
            con.commit()
        log.info("Cleared fundamentals cache — re-fetching all")
        run_fundamentals_update(stocks, workers=2)


if __name__ == "__main__":
    main()
