"""
Inference for trading results where observations are strongly correlated.

Fills within a day are not independent, so naive t-stats can overstate significance several-fold.

ratio_t and ratio_diff_t: cluster-robust t-stats for per-share ratios such as P&L per contract, clustered by day or market.

wild_bounds: wild cluster bootstrap bounds on daily means. Every sign flip is enumerated up to 10 days, and Webb weights are used beyond that.

verdict: the pass rule we fixed in advance and used for every strategy result. A POSITIVE verdict needs a lower bound above zero that survives dropping any single day, and dropping the two best days.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

MIN_DAYS = 8
ENUMERATE_UP_TO = 10
B_DRAWS = 4000
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def ratio_t(num, den, groups) -> tuple[float, float, int]:
    num, den = np.asarray(num, float), np.asarray(den, float)
    if len(num) == 0 or den.sum() <= 0:
        return np.nan, np.nan, 0
    g = pd.Series(np.asarray(groups)).factorize()[0]
    a, w = np.bincount(g, weights=num), np.bincount(g, weights=den)
    theta = a.sum() / w.sum()
    n = len(a)
    if n < 3:
        return float(theta), np.nan, n
    u = a - theta * w
    se = np.sqrt(np.sum(u**2) * n / (n - 1)) / w.sum()
    return float(theta), float(theta / se) if se > 0 else np.nan, n


def ratio_diff_t(num1, den1, g1, num2, den2, g2) -> tuple[float, float]:
    num1, den1, num2, den2 = (np.asarray(x, float) for x in (num1, den1, num2, den2))
    if len(num1) == 0 or len(num2) == 0 or den1.sum() <= 0 or den2.sum() <= 0:
        return np.nan, np.nan
    days = sorted(set(g1) | set(g2))
    idx = {d: i for i, d in enumerate(days)}
    i1, i2 = np.array([idx[d] for d in g1]), np.array([idx[d] for d in g2])
    a1, w1 = np.bincount(i1, weights=num1, minlength=len(days)), np.bincount(
        i1, weights=den1, minlength=len(days)
    )
    a2, w2 = np.bincount(i2, weights=num2, minlength=len(days)), np.bincount(
        i2, weights=den2, minlength=len(days)
    )
    t1, t2 = a1.sum() / w1.sum(), a2.sum() / w2.sum()
    u = (a1 - t1 * w1) / w1.sum() - (a2 - t2 * w2) / w2.sum()
    n = len(days)
    if n < 3:
        return float(t1 - t2), np.nan
    se = np.sqrt(np.sum(u**2) * n / (n - 1))
    return float(t1 - t2), float((t1 - t2) / se) if se > 0 else np.nan


def wild_bounds(
    daily: np.ndarray, *, alpha: float = 0.05, seed: int = 20261105, b: int = B_DRAWS
) -> tuple[float, float, float]:
    d = np.asarray(daily, float)
    g = len(d)
    if g == 0:
        return np.nan, np.nan, np.nan
    m = d.mean()
    e = d - m
    if g <= ENUMERATE_UP_TO:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=g)))
        dev = signs @ e / g
    else:
        rng = np.random.default_rng(seed)
        w = rng.choice(WEBB, size=(b, g))
        dev = w @ e / g
    hi_dev = np.quantile(dev, 1 - alpha)
    lo_dev = np.quantile(dev, alpha)
    return float(m), float(m - hi_dev), float(m - lo_dev)


def drop_days_rule(daily: pd.Series, **kw) -> tuple[bool, dict]:
    d = daily.astype(float)
    if len(d) < 3:
        return False, {"reason": "too few days"}
    worst = None
    for day in d.index:
        _, lo, _ = wild_bounds(d.drop(day).to_numpy(), **kw)
        if worst is None or lo < worst[1]:
            worst = (str(day), lo)
    top2 = d.sort_values(ascending=False).index[:2]
    _, lo2, _ = wild_bounds(d.drop(top2).to_numpy(), **kw)
    ok = worst[1] > 0 and lo2 > 0
    return bool(ok), {
        "min_lower_drop_one": worst[1],
        "drop_one_day": worst[0],
        "lower_drop_top2": lo2,
        "top2_days": [str(x) for x in top2],
    }


def verdict(daily: pd.Series, **kw) -> dict:
    # verdict on a series of daily p&l sums indexed by day
    d = daily.dropna().astype(float)
    out = {
        "days": int(len(d)),
        "mean_per_day": float(d.mean()) if len(d) else np.nan,
        "total": float(d.sum()) if len(d) else np.nan,
        "worst_day": float(d.min()) if len(d) else np.nan,
        "ex_best_day": float(d.sum() - d.max()) if len(d) else np.nan,
        "best_day_share": (
            float(d.max() / abs(d.sum())) if len(d) and d.sum() != 0 else np.nan
        ),
    }
    if len(d) >= 3:
        m, lo, hi = wild_bounds(d.to_numpy(), **kw)
        sd = d.std(ddof=1)
        out.update(
            lower_95=lo,
            upper_95=hi,
            t_daily=float(m / (sd / np.sqrt(len(d)))) if sd > 0 else np.nan,
        )
    else:
        out.update(lower_95=np.nan, upper_95=np.nan, t_daily=np.nan)
    if len(d) < MIN_DAYS:
        out.update(verdict="INCONCLUSIVE", why=f"fewer than {MIN_DAYS} days with fills")
        return out
    ok, info = drop_days_rule(d, **kw)
    out.update(info)
    if out["lower_95"] > 0 and ok:
        out.update(
            verdict="POSITIVE", why="lower bound > 0 and the drop-days rule passes"
        )
    elif out["upper_95"] < 0:
        out.update(verdict="NEGATIVE", why="upper bound < 0")
    elif out["lower_95"] > 0:
        out.update(
            verdict="INCONCLUSIVE", why="lower bound > 0 but dropping days breaks it"
        )
    else:
        out.update(verdict="INCONCLUSIVE", why="bounds straddle zero")
    return out
