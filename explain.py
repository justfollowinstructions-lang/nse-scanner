#!/usr/bin/env python3
"""
Explain — rule-based, per-signal prose generator for the HTML dashboard
=========================================================================
Ported from the cup_test scanner's explain.py. Converts one signal row
(a dict — a `signals` table record, or the equivalent df.iterrows()
Series.to_dict()) into human-readable prose: why the scanner flagged
this stock, what's attractive, what's risky, whether buying is
recommended, and a rule-by-rule PASS/FAIL/WARN breakdown.

cup_test's version traced every sentence back to Cup&Handle geometry
fields (cup depth, handle quality, rim symmetry). nse-scanner has no
single geometry model — it has 14 different pattern detectors — so this
version traces every sentence back to the SAME fields nse-scanner's own
score10 composite already uses (CANSLIM, RS percentile, risk:reward,
volume surge, TI65, Lynch score, Weinstein stage, volume dry-up,
Piotroski F-score, promoter pledge, bulk/block deals). That keeps the
same design principle: nothing here is invented — if a field is
missing, the corresponding sentence is simply omitted.
"""

from __future__ import annotations

import math


def _f(row: dict, key: str, default=None):
    """Safe field getter — treats NaN/None uniformly as missing."""
    val = row.get(key, default)
    if val is None:
        return default
    try:
        if isinstance(val, float) and math.isnan(val):
            return default
    except Exception:
        pass
    return val


def _money(val) -> str:
    if val is None:
        return "—"
    return f"₹{val:,.2f}"


# ─── Overall rating ─────────────────────────────────────────────────────────

def overall_rating(row: dict) -> dict:
    """Returns {"label", "color", "reason"} — the top-line verdict."""
    tier = str(_f(row, "tier", "") or "")
    score10 = _f(row, "score10", 0) or 0
    hc = bool(row.get("_is_high_conviction"))

    if hc:
        return {
            "label": "Very High Quality",
            "color": "green",
            "reason": f"Score {score10:.1f}/10 with BUY STRONG tier, risk:reward, "
                      f"volume confirmation, and RS leadership all aligned — the "
                      f"scanner's highest-conviction category across all 14 "
                      f"pattern types.",
        }
    if "BUY STRONG" in tier:
        return {
            "label": "Strong Buy",
            "color": "green",
            "reason": f"Score {score10:.1f}/10 — CANSLIM, RS percentile, risk:reward "
                      f"and volume confirmation are strongly aligned.",
        }
    if "BUY MODERATE" in tier:
        return {
            "label": "Good Candidate",
            "color": "blue",
            "reason": f"Score {score10:.1f}/10 — a solid setup with most factors "
                      f"confirming, though not as complete as a Strong Buy.",
        }
    if "WATCH" in tier:
        return {
            "label": "Watch",
            "color": "yellow",
            "reason": f"Score {score10:.1f}/10 — pattern detected but several "
                      f"confirming factors are missing; worth monitoring rather "
                      f"than acting on immediately.",
        }
    return {
        "label": "Weak / Avoid",
        "color": "red",
        "reason": f"Score {score10:.1f}/10 — too few confirming factors align; "
                  f"the scanner's own rules treat this as a pass.",
    }


# ─── "Why did the scanner detect this?" — rule PASS/FAIL/WARN list ─────────

def scanner_reasons(row: dict) -> list[dict]:
    reasons = []

    def add(rule, status, detail):
        reasons.append({"rule": rule, "status": status, "detail": detail})

    cs = _f(row, "canslim_score")
    dc = _f(row, "data_completeness")
    if cs is not None and dc:
        pct = 100 * cs / max(dc, 1)
        add("CANSLIM checklist", "pass" if pct >= 60 else "warn" if pct >= 35 else "fail",
            f"{cs:.0f}/{dc:.0f} CANSLIM sub-rules passed ({pct:.0f}%).")

    rs = _f(row, "rs_percentile")
    if rs is not None:
        add("Relative Strength", "pass" if rs >= 80 else "warn" if rs >= 60 else "fail",
            f"RS Percentile {rs:.0f} — true cross-sectional ranking against the "
            f"whole NSE universe scanned today, not just the index.")

    rr = _f(row, "risk_reward")
    if rr is not None:
        add("Risk:Reward", "pass" if rr >= 2.0 else "warn" if rr >= 1.5 else "fail",
            f"{rr:.2f}:1 measured to the first target.")

    vs = _f(row, "vol_surge")
    if vs is not None:
        add("Volume confirmation", "pass" if vs >= 1.4 else "warn" if vs >= 1.0 else "fail",
            f"Volume is {vs:.2f}× its recent average" +
            (" — real participation behind the move." if vs >= 1.4 else
             " — O'Neil's rule wants ≥1.4×; below that, a move is more likely "
             "to fail or fade."))

    stage = str(_f(row, "stage", "") or "")
    if stage:
        add("Weinstein Stage", "pass" if "Stage2" in stage else "warn" if "Stage1" in stage else "fail",
            f"Classified as {stage} — Stage 2 (markup) is the only stage "
            f"Weinstein's method considers safe to buy in.")

    if row.get("vol_dryup"):
        add("Volume dry-up before signal", "pass",
            "Volume contracted into the setup — a classic 'smart money "
            "stealth accumulation' tell before a genuine move.")

    ti = _f(row, "ti65")
    if ti is not None:
        add("Trend Intensity (TI65)", "pass" if ti >= 1.05 else "warn",
            f"TI65 = {ti:.2f} — price vs its 65-period trend envelope.")

    ls = _f(row, "lynch_score_val")
    if ls is not None:
        add("Peter Lynch checklist", "pass" if ls >= 4 else "warn" if ls >= 2 else "fail",
            f"{ls:.0f}/6 Lynch-style quality checks passed.")

    if row.get("earnings_near"):
        add("Earnings safety", "warn",
            "An earnings event falls within the next 14 days — position "
            "sizing/timing risk is elevated until it's past.")
    else:
        add("Earnings safety", "pass", "No earnings event within the next 14 days.")

    pio = _f(row, "piotroski_score")
    if pio is not None:
        add("Piotroski F-Score", "pass" if pio >= 7 else "warn" if pio >= 4 else "fail",
            f"{pio:.0f}/9 — financial-statement strength checklist "
            f"(profitability, leverage, efficiency).")

    pledge = _f(row, "pledge_pct")
    if pledge is not None:
        add("Promoter pledge", "pass" if pledge < 5 else "warn" if pledge <= 20 else "fail",
            f"{pledge:.1f}% of promoter holding is pledged." +
            (" Above 20% is a real red flag." if pledge > 20 else ""))

    bulk = _f(row, "bulk_deal_cr")
    if bulk:
        add("Bulk/block deal today", "pass",
            f"₹{bulk:.1f} Cr in bulk/block deal activity recorded — "
            f"institutional footprint on the signal day.")

    converging = row.get("converging")
    if converging and str(converging).strip() not in ("", "nan", "None"):
        add("Multi-pattern confluence", "pass",
            f"Also flagged by: {converging} — several independent detectors "
            f"agreeing raises confidence this isn't a single noisy signal.")

    return reasons


# ─── "Why should I buy this?" ───────────────────────────────────────────────

def why_buy(row: dict) -> dict:
    rating = overall_rating(row)
    tier = str(_f(row, "tier", "") or "")
    rs = _f(row, "rs_percentile")
    vs = _f(row, "vol_surge")
    rr = _f(row, "risk_reward")
    stage = str(_f(row, "stage", "") or "")

    recommend = "BUY" in tier and (rr or 0) >= 1.5

    paragraphs = []

    if "Stage2" in stage:
        paragraphs.append(
            "Trend: this stock is in a Weinstein Stage 2 advance — the only "
            "stage his method treats as safe to buy, since price, volume and "
            "moving averages are all pointed the same way."
        )
    elif "Stage1" in stage:
        paragraphs.append(
            "Trend: this stock is still in a Stage 1 base — not yet confirmed "
            "as trending, so this reads as an early, higher-risk entry rather "
            "than a confirmed breakout continuation."
        )
    elif "Stage3" in stage or "Stage4" in stage:
        paragraphs.append(
            "Trend: this stock is showing Stage 3/4 characteristics (topping "
            "or declining) — a genuine concern that works against the pattern "
            "signal, size accordingly."
        )

    if rs is not None:
        if rs >= 90:
            paragraphs.append(
                f"Momentum: RS Percentile {rs:.0f} marks this as a genuine "
                f"relative-strength leader against every other stock scanned "
                f"today, not just the index."
            )
        elif rs >= 70:
            paragraphs.append(
                f"Momentum: RS Percentile {rs:.0f} shows real relative strength "
                f"without yet being an outright leader."
            )
        else:
            paragraphs.append(
                f"Momentum: RS Percentile {rs:.0f} is on the weaker side — this "
                f"stock hasn't been a relative-strength leader recently, which "
                f"lowers the odds of a fast, powerful move even if the pattern "
                f"looks clean."
            )

    if vs is not None:
        if vs >= 1.4:
            paragraphs.append(
                f"Volume: {vs:.2f}× average — real participation behind the "
                f"move, the kind of confirmation institutional buying tends to "
                f"leave behind."
            )
        else:
            paragraphs.append(
                f"Volume: {vs:.2f}× average — present but not surging. Worth "
                f"waiting for a stronger volume day before treating this as "
                f"confirmed rather than tentative."
            )

    if rr is not None:
        if rr >= 2.0:
            paragraphs.append(
                f"Risk:Reward is {rr:.2f}:1 to the first target — a favourable "
                f"asymmetry even allowing for some signals not working out."
            )
        else:
            paragraphs.append(
                f"Risk:Reward is only {rr:.2f}:1 — below the 2:1 the strategy "
                f"prefers. Size carefully if taking this trade."
            )

    notes = str(_f(row, "notes", "") or "")
    if "RS-LEAD" in notes:
        paragraphs.append(
            "RS Line Leadership: this stock's relative-strength line is "
            "hitting new highs alongside price — O'Neil considered this one "
            "of the single best confirming signals available."
        )

    if not paragraphs:
        paragraphs.append(
            "Limited data available to build a full case either way — treat "
            "this as a pattern-only signal and verify manually before acting."
        )

    return {"recommend": recommend, "paragraphs": paragraphs}


# ─── Full explanation block (strengths / weaknesses / conclusion) ─────────

def full_explanation(row: dict) -> dict:
    pattern = _f(row, "pattern", "pattern")
    tier = str(_f(row, "tier", "") or "")
    score10 = _f(row, "score10", 0) or 0
    rr = _f(row, "risk_reward")
    rs = _f(row, "rs_percentile")
    vs = _f(row, "vol_surge")
    pio = _f(row, "piotroski_score")
    pledge = _f(row, "pledge_pct")

    why_detected = (
        f"The scanner's {pattern} detector flagged this stock, and the "
        f"composite scoring system rates it {score10:.1f}/10 ({tier.strip()})."
    )

    strengths, weaknesses = [], []
    if rs is not None:
        (strengths if rs >= 70 else weaknesses).append(
            f"RS Percentile {rs:.0f}" + (" — a genuine leader." if rs >= 70 else " — lagging the broader market.")
        )
    if vs is not None:
        (strengths if vs >= 1.4 else weaknesses).append(
            f"Volume surge {vs:.2f}x" + (" confirms real buying interest." if vs >= 1.4 else " is unconvincing on its own.")
        )
    if rr is not None:
        (strengths if rr >= 2.0 else weaknesses).append(
            f"Risk:Reward {rr:.2f}:1" + (" is a favourable setup." if rr >= 2.0 else " is thin — size down.")
        )
    if pio is not None:
        (strengths if pio >= 7 else weaknesses).append(
            f"Piotroski F-Score {pio:.0f}/9" + (" — strong fundamentals." if pio >= 7 else " — fundamentals need scrutiny.")
        )
    if pledge is not None and pledge > 10:
        weaknesses.append(f"Promoter pledge {pledge:.1f}% is elevated — a governance risk to watch.")

    if not strengths:
        strengths.append("No standout strength flagged by available data — treat as a pattern-only signal.")
    if not weaknesses:
        weaknesses.append("No major weakness flagged by available data.")

    institutional_note = (
        f"Bulk/block deal value today: ₹{_f(row,'bulk_deal_cr',0) or 0:.1f} Cr. "
        f"Promoter pledge: {pledge:.1f}%." if pledge is not None else
        "No promoter pledge data available for this stock."
    )

    if "BUY" in tier:
        conclusion = (
            f"On balance the scanner rates this a {tier.strip().lower()} setup — "
            f"still verify the chart yourself and respect the stop-loss level "
            f"the scanner computed."
        )
    else:
        conclusion = (
            "On balance this doesn't clear the scanner's own bar for a buy "
            "recommendation yet — better suited to the watchlist than an "
            "immediate entry."
        )

    return {
        "why_detected": why_detected,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "institutional_note": institutional_note,
        "conclusion": conclusion,
    }


def sell_rules_text(row: dict) -> str:
    """Generic-but-specific sell checklist using this stock's own numbers."""
    stop = row.get("stop_loss")
    t1 = row.get("target_1")
    t2 = row.get("target_2")
    parts = []
    if stop:
        parts.append(f"Exit fully if price closes below {_money(stop)} — the "
                      f"scanner's computed stop-loss for this setup.")
    if t1:
        parts.append(f"Consider trimming a third of the position at Target 1 "
                      f"({_money(t1)}) to lock in partial gains.")
    if t2:
        parts.append(f"Trail the stop up to breakeven once Target 2 "
                      f"({_money(t2)}) is reached.")
    parts.append("Re-check the pattern's premise (volume, RS, stage) on any "
                  "sharp adverse move — don't wait for the hard stop alone.")
    return "||".join(parts)
