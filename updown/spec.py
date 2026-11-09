"""
Turns market titles and rules text into contract specs.

A title like "Bitcoin Up or Down - October 24, 2:30AM-2:45AM ET", together with rules that mention the Chainlink BTC/USD TWAP data stream, becomes a Spec. The Spec holds the payoff kind, the underlying, the window, and whether settlement is on a point price or a TWAP.

The payoff kinds are up or down, above, below, between, and touch.

settle reproduces the official outcome from 1-minute candles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
SYMBOLS = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "solana": "SOLUSDT",
    "xrp": "XRPUSDT",
}
MONTHS = "january february march april may june july august september october november december".split()
_MON = "(" + "|".join(MONTHS) + ")"
_NUM = r"\$?([\d,]+(?:\.\d+)?)"
_CLOCK = r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM)"
CHAINLINK_TWAP_S = 60


@dataclass(frozen=True)
class Spec:
    kind: str
    symbol: str
    end: pd.Timestamp  # resolution instant in utc
    start: pd.Timestamp | None = None  # window start for touch or base instant for up or down
    lo: float | None = None  # strike for above, below and touch or the lower bracket
    hi: float | None = None  # upper bracket for between
    ref_offset: int = -1
    twap_s: int = 0


def _f(x: str) -> float:
    return float(x.replace(",", ""))


def _clock_min(h: str, m: str | None, ap: str) -> int:
    return (int(h) % 12 + (12 if ap.upper() == "PM" else 0)) * 60 + int(m or 0)


def _et_midnight(year: int, mon: str, day: str) -> pd.Timestamp:
    return pd.Timestamp(
        datetime(year, MONTHS.index(mon) + 1, int(day), tzinfo=ET)
    ).tz_convert("UTC")


def parse_spec(
    title: str, close_ts, open_ts=None, rules: str | None = None
) -> Spec | None:
    if not title or close_ts is None or pd.isna(close_ts):
        return None
    t = title.strip()
    tl = t.lower()
    m = re.search(r"\b(bitcoin|ethereum|solana|xrp)\b", tl)
    if not m:
        return None
    sym = SYMBOLS[m.group(1)]
    end = (
        pd.Timestamp(close_ts).tz_convert("UTC")
        if pd.Timestamp(close_ts).tzinfo
        else pd.Timestamp(close_ts, tz="UTC")
    )
    end_et = end.tz_convert(ET)
    end_clock = end_et.hour * 60 + end_et.minute

    if "up or down" in tl:
        w = re.search(rf"{_MON} \d+, {_CLOCK}\s*-\s*{_CLOCK} et", tl, re.I)
        if w:
            a = _clock_min(w.group(2), w.group(3), w.group(4))
            b = _clock_min(w.group(5), w.group(6), w.group(7))
            if b != end_clock:
                return None
            twap = 0 if rules and "twap" not in rules.lower() else CHAINLINK_TWAP_S
            # 5 minute markets name their own stream in the rules like twap-30s so use that length
            if twap and rules and (m := re.search(r"twap-(\d+)s", rules, re.I)):
                twap = int(m.group(1))
            return Spec(
                "updown_cl",
                sym,
                end,
                end - timedelta(minutes=(b - a) % 1440 or 1440),
                twap_s=twap,
            )
        h = re.search(rf"{_MON} \d+, {_CLOCK} et", tl, re.I)
        if h:
            if (
                _clock_min(h.group(2), h.group(3), h.group(4)) + 60
            ) % 1440 != end_clock:
                return None
            return Spec("updown", sym, end, end - timedelta(hours=1))
        if re.search(rf"up or down on {_MON} \d+", tl) and end_clock == 12 * 60:
            return Spec("updown", sym, end, end - timedelta(days=1), ref_offset=0)
        return None

    nums = [_f(x) for x in re.findall(r"\$([\d,]+(?:\.\d+)?)", t)]
    hourly = re.search(rf"\babove {_NUM} on {_MON} \d+, {_CLOCK} et", tl, re.I)
    if hourly:
        if _clock_min(hourly.group(3), hourly.group(4), hourly.group(5)) != end_clock:
            return None
        return Spec("above", sym, end, lo=_f(hourly.group(1)))
    if re.search(r"will the price of .+ be (above|greater than|less than|between)", tl):
        if end_clock != 12 * 60 or not nums:
            return None
        if "between" in tl:
            if len(nums) < 2:
                return None
            lo, hi = sorted(nums[:2])
            return Spec("between", sym, end, lo=lo, hi=hi)
        return Spec("below" if "less than" in tl else "above", sym, end, lo=nums[0])

    touch = re.search(r"\b(reach|hit|dip to)\b", tl)
    if touch and nums and end_clock == 0:
        kind = "touch_down" if touch.group(1) == "dip to" else "touch_up"
        start = _touch_start(tl, end, open_ts, rules)
        if (
            start is not None
            and rules
            and "creation of this market" in rules.lower()
            and open_ts is not None
            and not pd.isna(open_ts)
        ):
            start = max(start, pd.Timestamp(open_ts).tz_convert("UTC").floor("min"))
        if start is None or start >= end:
            return None
        return Spec(kind, sym, end, start, lo=nums[0])
    return None


def _touch_start(
    tl: str, end: pd.Timestamp, open_ts, rules: str | None
) -> pd.Timestamp | None:
    last_day = (end - timedelta(minutes=1)).tz_convert(ET)
    y = last_day.year
    if re.search(rf"\bon {_MON} \d+\?", tl):
        return end - timedelta(days=1) if end.tz_convert(ET).hour == 0 else None
    r = re.search(rf"\$[\d,.]+ {_MON} (\d+)\s*-\s*(?:{_MON} )?(\d+)\?", tl)
    if r:
        s = _et_midnight(y, r.group(1), r.group(2))
        return s if s < end else _et_midnight(y - 1, r.group(1), r.group(2))
    r = re.search(rf"\bin {_MON}\?", tl)
    if r:
        return _et_midnight(y, r.group(1), "1")
    if re.search(r"\bby ", tl):
        if rules:
            g = re.search(
                rf"between {_MON} (\d+), (\d{{4}}),? (\d{{1,2}}):(\d{{2}})",
                rules.lower(),
            )
            if g:
                return _et_midnight(
                    int(g.group(3)), g.group(1), g.group(2)
                ) + timedelta(hours=int(g.group(4)), minutes=int(g.group(5)))
        if open_ts is not None and not pd.isna(open_ts):
            return pd.Timestamp(open_ts).tz_convert("UTC").floor("min")
    return None


def settle(spec: Spec, bars: pd.DataFrame) -> float:
    def ref(instant: pd.Timestamp) -> float:
        k = instant + timedelta(minutes=spec.ref_offset)
        return float(bars.at[k, "close"]) if k in bars.index else np.nan

    if spec.kind in ("above", "below", "between"):
        p = ref(spec.end)
        if np.isnan(p):
            return np.nan
        if spec.kind == "above":
            return float(p > spec.lo)
        if spec.kind == "below":
            return float(p < spec.lo)
        return float(spec.lo <= p < spec.hi)
    if spec.kind == "updown":
        daily = spec.end - spec.start == timedelta(days=1)
        if daily:  # 12:00 ET close vs the day before and a tie resolves 50 50
            a, b = ref(spec.start), ref(spec.end)
        else:  # 1 hour candle so open of its first minute vs close of its last minute
            a = (
                float(bars.at[spec.start, "open"])
                if spec.start in bars.index
                else np.nan
            )
            b = ref(spec.end)
        if np.isnan(a) or np.isnan(b):
            return np.nan
        if daily and a == b:
            return 0.5
        return float(b > a) if daily else float(b >= a)
    w = bars.loc[(bars.index >= spec.start) & (bars.index < spec.end)]
    n_exp = int((spec.end - spec.start) / timedelta(minutes=1))
    if len(w) < 0.99 * n_exp:
        return np.nan
    if spec.kind == "updown_cl":
        return float(w["close"].iloc[-1] >= w["open"].iloc[0])
    if spec.kind == "touch_up":
        return float(w["high"].max() >= spec.lo)
    if spec.kind == "touch_down":
        return float(w["low"].min() <= spec.lo)
    return np.nan
