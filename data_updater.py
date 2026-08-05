#!/usr/bin/env python3
"""
Batch Downloader — resilient multi-symbol OHLCV download mechanism
=====================================================================
Ported from the `cup_test` scanner's `downloader.py` ("this is what makes
download once, update daily work" / "doesn't fail this lot"). Adapted so
it writes into nse-scanner's EXISTING SQLite price_cache — every reader
(_cache_read, read_cache in scanner.py, read_cache/write_cache in
data_updater.py) is untouched. Only the *download* mechanism changes.

Why this replaces the old per-symbol loop
------------------------------------------
scanner.py's warm_cache() and data_updater.py's run_eod_update() both
issued ONE yf.download() call PER SYMBOL, fanned out across a small
thread pool (workers=4-8 in scanner.py; data_updater.py gave up on
parallelism entirely — MAX_WORKERS=1 — specifically because parallel
per-symbol calls tripped Yahoo's rate limiter too often on ~2000+ NSE
symbols/day). That's reliable but slow (45-60 min for a full run) and
still fails a meaningful slice of symbols on a bad day.

cup_test's downloader.py proved a batch approach at the same universe
scale: ~50 symbols per yf.download() call = 1 HTTP round-trip instead of
50. Fewer round-trips means far fewer chances to get rate-limited, and
failures are triaged into two queues (fast retry for timeouts, slow
escalating-backoff retry for 429s) instead of one blunt retry-with-sleep
loop. That combination is what "doesn't fail this lot" refers to.

Usage
-----
    from batch_downloader import run_batch_download

    stats = run_batch_download(
        symbols=["RELIANCE.NS", "TCS.NS", ...],
        get_last_date=lambda sym: my_cache_last_date(sym),   # None => full history
        save_df=lambda sym, df: my_cache_write(sym, df),
        interval="1d",
        full_history_start="2015-01-01",
        log=log,
    )

`get_last_date` / `save_df` are callbacks so this module knows nothing
about SQLite schemas, table names, or timeframe columns — the caller
(scanner.py's warm_cache, data_updater.py's run_eod_update) owns that.
"""

from __future__ import annotations

import random
import time
from datetime import date, timedelta
from typing import Callable, Optional

import pandas as pd
import yfinance as yf

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# ─── Tuneables (same defaults cup_test proved out at full-universe scale) ──
BATCH_SIZE               = 50
BATCH_DELAY_SECONDS      = 3
MAX_RETRIES              = 5      # rate-limit retry rounds
RATELIMIT_RETRY_WAIT_MIN = 2       # minutes, base for exponential backoff
EXPONENTIAL_BASE         = 2.0
TIMEOUT_RETRY_WAIT_SEC   = 30
DEFAULT_FULL_HISTORY_START = "2015-01-01"


def run_batch_download(
    symbols: list[str],
    get_last_date: Callable[[str], Optional[str]],
    save_df: Callable[[str, pd.DataFrame], None],
    interval: str = "1d",
    full_history_start: str = DEFAULT_FULL_HISTORY_START,
    full_refresh: bool = False,
    batch_size: int = BATCH_SIZE,
    log=None,
    shuffle: bool = True,
) -> dict:
    """
    Main entry point. Downloads/updates OHLCV for every symbol via batched
    yf.download() calls, retrying failures through timeout / rate-limit
    queues. Returns a stats dict: {saved, failed, batches, elapsed_sec}.

    get_last_date(sym) -> "YYYY-MM-DD" | None
        Last cached bar date for this symbol+interval. None (or
        full_refresh=True) means "download full history from
        full_history_start".

    save_df(sym, df) -> None
        Persist new bars for this symbol. df has columns
        Open/High/Low/Close/Volume and a DatetimeIndex, restricted to
        bars strictly after the symbol's last cached date. Caller
        decides how to merge/upsert.
    """
    log = log or _NullLog()
    t0 = time.time()

    targets = list(symbols)
    if shuffle:
        random.shuffle(targets)   # break alphabetical clustering across batches

    saved = 0
    failed: list[str] = []
    batches = _make_batches(targets, batch_size)
    log.info(f"Batch download: {len(targets)} symbols ({interval}) in "
             f"{len(batches)} batches of {batch_size}")

    timeout_queue: list[str] = []
    ratelimit_queue: list[str] = []

    for i, batch in enumerate(batches, 1):
        n_saved, timed_out, rate_limited = _download_batch(
            batch, get_last_date, save_df, interval,
            full_history_start, full_refresh, log,
        )
        saved += n_saved
        timeout_queue.extend(timed_out)
        ratelimit_queue.extend(rate_limited)
        if i % 10 == 0 or i == len(batches):
            log.info(f"  Batch {i}/{len(batches)} — saved={saved} "
                     f"timeouts_q={len(timeout_queue)} ratelimit_q={len(ratelimit_queue)}")
        if i < len(batches):
            time.sleep(BATCH_DELAY_SECONDS)

    if timeout_queue:
        log.info(f"Retrying {len(timeout_queue)} timeout symbols after "
                 f"{TIMEOUT_RETRY_WAIT_SEC}s ...")
        time.sleep(TIMEOUT_RETRY_WAIT_SEC)
        still_failed = _retry_individually(
            timeout_queue, get_last_date, save_df, interval,
            full_history_start, full_refresh, log, max_attempts=3, wait_sec=15,
        )
        saved += len(timeout_queue) - len(still_failed)
        failed.extend(still_failed)

    if ratelimit_queue:
        recovered, still_bad = _retry_ratelimited(
            ratelimit_queue, get_last_date, save_df, interval,
            full_history_start, full_refresh, log,
        )
        saved += recovered
        failed.extend(still_bad)

    elapsed = time.time() - t0
    log.info(f"Batch download done: {elapsed:.0f}s | saved={saved} | "
             f"permanently_failed={len(failed)}")
    return {"saved": saved, "failed": failed, "batches": len(batches),
            "elapsed_sec": round(elapsed, 1)}


# ─── Batch internals ───────────────────────────────────────────────────────

def _download_batch(batch, get_last_date, save_df, interval,
                     full_history_start, full_refresh, log
                     ) -> tuple[int, list[str], list[str]]:
    """Download one batch. Returns (n_saved, timed_out_syms, rate_limited_syms)."""
    starts = {}
    for sym in batch:
        if full_refresh:
            starts[sym] = full_history_start
            continue
        last = None
        try:
            last = get_last_date(sym)
        except Exception:
            last = None
        starts[sym] = _next_day(last) if last else full_history_start

    end = (date.today() + timedelta(days=1)).isoformat()
    min_start = min(starts.values())

    saved = 0
    timed_out: list[str] = []
    rate_limited: list[str] = []

    try:
        raw = yf.download(
            batch, start=min_start, end=end, interval=interval,
            auto_adjust=True, progress=False, threads=False,
        )
    except Exception as exc:
        err = str(exc).lower()
        if "timeout" in err or "timed out" in err or "curl: (28)" in err:
            log.debug(f"Batch timeout — queuing for fast retry: {batch[:3]}...")
            return 0, batch, []
        if "429" in err or "rate" in err or "too many" in err:
            log.debug("Rate limit hit on batch — queuing for slow retry")
            return 0, [], batch
        log.debug(f"Batch exception: {exc}")
        return 0, batch, []

    if raw is None or raw.empty:
        return 0, batch, []

    for sym in batch:
        df = _extract_symbol_robust(raw, sym, len(batch))
        if df is None or df.empty:
            timed_out.append(sym)
            continue
        sym_start = starts[sym]
        df = df[df.index >= pd.Timestamp(sym_start)]
        if df.empty:
            saved += 1   # already up to date — not a failure
            continue
        try:
            save_df(sym, df)
            saved += 1
        except Exception as e:
            log.debug(f"save_df failed for {sym}: {e}")
            timed_out.append(sym)

    return saved, timed_out, rate_limited


def _extract_symbol_robust(raw: pd.DataFrame, symbol: str, n_syms: int) -> Optional[pd.DataFrame]:
    """Robustly extract single-symbol OHLCV from a yfinance batch response,
    handling both the flat-column (n=1) and MultiIndex (n>1) shapes."""
    try:
        if n_syms == 1 or raw.columns.nlevels == 1:
            df = raw.copy()
        else:
            if symbol not in raw.columns.get_level_values(1):
                return None
            df = raw.xs(symbol, axis=1, level=1).copy()

        df.columns = [str(c).strip() for c in df.columns]
        if "Adj Close" in df.columns and "Close" not in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        elif "Adj Close" in df.columns:
            df = df.drop(columns=["Adj Close"])

        present = [c for c in OHLCV_COLS if c in df.columns]
        if len(present) < 4:
            return None
        df = df[present].dropna(how="all")
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if "Close" in df.columns:
            df = df[df["Close"] > 0]
        return df if not df.empty else None
    except Exception:
        return None


# ─── Single-symbol fallback (benchmarks, final retries) ───────────────────

def _download_single_with_retry(symbol, get_last_date, save_df, interval,
                                 full_history_start, full_refresh, log,
                                 attempts: int = 3) -> bool:
    last = None if full_refresh else get_last_date(symbol)
    sym_start = _next_day(last) if last else full_history_start
    for attempt in range(1, attempts + 1):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=sym_start,
                end=(date.today() + timedelta(days=1)).isoformat(),
                interval=interval,
                auto_adjust=True,
            )
            if df.empty:
                time.sleep(5 * attempt)
                continue
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.columns = [str(c) for c in df.columns]
            present = [c for c in OHLCV_COLS if c in df.columns]
            df = df[present].dropna(how="all")
            if "Close" in df.columns:
                df = df[df["Close"] > 0]
            if df.empty:
                return True   # nothing new — not a failure
            save_df(symbol, df)
            return True
        except Exception as e:
            log.debug(f"{symbol} single-retry attempt {attempt}: {e}")
            time.sleep(10 * attempt)
    return False


def _retry_individually(symbols, get_last_date, save_df, interval,
                         full_history_start, full_refresh, log,
                         max_attempts: int, wait_sec: int) -> list[str]:
    still_failed = []
    for sym in symbols:
        ok = _download_single_with_retry(
            sym, get_last_date, save_df, interval,
            full_history_start, full_refresh, log, attempts=max_attempts,
        )
        if not ok:
            still_failed.append(sym)
        time.sleep(wait_sec)
    return still_failed


def _retry_ratelimited(symbols, get_last_date, save_df, interval,
                        full_history_start, full_refresh, log
                        ) -> tuple[int, list[str]]:
    remaining = list(symbols)
    recovered = 0
    for attempt in range(1, MAX_RETRIES + 1):
        if not remaining:
            break
        wait = RATELIMIT_RETRY_WAIT_MIN * 60 * (EXPONENTIAL_BASE ** (attempt - 1))
        log.info(f"Rate-limit retry {attempt}/{MAX_RETRIES} — waiting {wait:.0f}s "
                 f"for {len(remaining)} symbols")
        time.sleep(wait)

        mini_batches = _make_batches(remaining, 10)
        still_bad = []
        for mb in mini_batches:
            n_saved, to, rl = _download_batch(
                mb, get_last_date, save_df, interval,
                full_history_start, full_refresh, log,
            )
            recovered += n_saved
            still_bad.extend(to + rl)
            time.sleep(5)

        log.info(f"Rate-limit retry {attempt}: {len(remaining) - len(still_bad)} "
                 f"recovered, {len(still_bad)} still failing")
        remaining = still_bad

    if remaining:
        log.warning(f"Permanently failed {len(remaining)} symbols after rate-limit "
                    f"retries: {remaining[:10]}")
    return recovered, remaining


# ─── Helpers ────────────────────────────────────────────────────────────────

def _next_day(date_str: str) -> str:
    try:
        d = pd.Timestamp(date_str).date()
        return (d + timedelta(days=1)).isoformat()
    except Exception:
        return DEFAULT_FULL_HISTORY_START


def _make_batches(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


class _NullLog:
    """Fallback no-op logger so this module works standalone without a
    logging.Logger passed in."""
    def info(self, *a, **k): pass
    def debug(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
