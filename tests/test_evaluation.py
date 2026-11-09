from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from updown import decompose as D
from updown import stats as S


def _fill(**kw) -> dict:
    base = {
        "side": "buy",
        "price": 0.53,
        "size": 10.0,
        "outcome": 1.0,
        "fee": 0.02,
        "theo_dec": 0.55,
        "theo_fill": 0.54,
        "mark_mid_60s": 0.58 - 0.53,
        "mark_source": "book",
        "pnl": 10 * (1 - 0.53) - 0.02,
    }
    base.update(kw)
    return base


def test_decomposition_reference_example_4_68():
    f = pd.DataFrame([_fill()])
    c = D.components(f).iloc[0]
    assert c["decision_edge"] == pytest.approx(0.20) and c[
        "decision_to_fill"
    ] == pytest.approx(-0.10)
    assert c["post_fill_markout"] == pytest.approx(0.40) and c[
        "settlement_residual"
    ] == pytest.approx(4.20)
    assert (
        c["fee_term"] == pytest.approx(-0.02)
        and c["total"] == pytest.approx(4.68)
        and not c["merged"]
    )
    assert D.check_identity(f) == (True, pytest.approx(0.0, abs=1e-12))


def test_decomposition_identity_random_rows_and_merging():
    rng = np.random.default_rng(1)
    n = 200
    side = np.where(rng.random(n) < 0.5, "buy", "sell")
    s = np.where(side == "buy", 1.0, -1.0)
    px, size, y = (
        rng.uniform(0.05, 0.95, n),
        rng.uniform(1, 50, n),
        rng.integers(0, 2, n).astype(float),
    )
    fee = rng.uniform(0, 0.1, n)
    f = pd.DataFrame(
        {
            "side": side,
            "price": px,
            "size": size,
            "outcome": y,
            "fee": fee,
            "theo_dec": rng.uniform(0.1, 0.9, n),
            "theo_fill": rng.uniform(0.1, 0.9, n),
            "mark_mid_60s": s * (rng.uniform(0.1, 0.9, n) - px),
            "mark_source": np.where(rng.random(n) < 0.3, None, "book"),
            "pnl": s * (y - px) * size - fee,
        }
    )
    f.loc[f["mark_source"].isna(), "mark_mid_60s"] = np.nan
    ok, d = D.check_identity(f)
    assert ok and d < 1e-9
    c = D.components(f)
    assert c["merged"].sum() == f["mark_source"].isna().sum()
    assert (c.loc[c["merged"], "post_fill_markout"] == 0).all()
    agg = D.aggregate(f)
    assert agg["total_c"].iloc[0] == pytest.approx(
        100 * f["pnl"].sum() / f["size"].sum()
    )


def test_clustered_t_is_smaller_than_naive_under_within_day_correlation():
    rng = np.random.default_rng(3)
    days = np.repeat(np.arange(20), 50)
    x = (
        rng.normal(0, 1, 20)[days] + rng.normal(0, 0.1, len(days)) + 0.3
    )  # a day effect dominates
    w = np.ones(len(x))
    m, t, n = S.ratio_t(x, w, days)
    naive = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert n == 20 and m == pytest.approx(x.mean()) and abs(t) < abs(naive) / 3
    d, td = S.ratio_diff_t(x, w, days, x, w, days)
    assert d == pytest.approx(0.0) and (np.isnan(td) or abs(td) < 1e-6)


def test_wild_bootstrap_enumeration_and_verdicts():
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m, lo, hi = S.wild_bounds(d)
    e = d - d.mean()
    devs = sorted(
        sum(s * ei for s, ei in zip(signs, e)) / len(d)
        for signs in __import__("itertools").product((-1, 1), repeat=len(d))
    )
    assert (
        m == pytest.approx(3.0)
        and lo == pytest.approx(3.0 - np.quantile(devs, 0.95))
        and hi == pytest.approx(3.0 - np.quantile(devs, 0.05))
    )
    m2, lo2, hi2 = S.wild_bounds(np.arange(1, 31, dtype=float))
    assert (
        lo2 < m2 < hi2 and lo2 == S.wild_bounds(np.arange(1, 31, dtype=float))[1]
    )  # webb weights with a fixed seed
    days = pd.date_range("2026-10-01", periods=20, freq="D").strftime("%Y-%m-%d")
    pos = pd.Series(np.full(20, 10.0) + np.arange(20) % 3, index=days)
    assert S.verdict(pos)["verdict"] == "POSITIVE"
    assert S.verdict(-pos)["verdict"] == "NEGATIVE"
    spike = pd.Series(
        np.r_[np.full(19, 0.5), 200.0], index=days
    )  # one day carries it so the bootstrap already straddles zero
    v = S.verdict(spike)
    assert v["verdict"] == "INCONCLUSIVE"
    rng = np.random.default_rng(11)
    few = pd.Series(
        np.r_[rng.normal(0, 0.3, 16), np.full(4, 20.0)], index=days
    )  # four days carry it
    v = S.verdict(few)
    assert (
        v["lower_95"] > 0 and v["verdict"] == "INCONCLUSIVE" and "drop" in v["why"]
    ), v
    assert S.verdict(pos.iloc[:5])["verdict"] == "INCONCLUSIVE"
    noise = pd.Series(
        np.array([1, -1, 2, -2, 3, -3, 1, -1, 2, -2], float), index=days[:10]
    )
    assert S.verdict(noise)["verdict"] == "INCONCLUSIVE"
