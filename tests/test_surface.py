from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from updown import surface as SF

U = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def test_implied_vol_round_trip():
    for cp, K in (("C", 110.0), ("P", 90.0), ("C", 100.0)):
        p = float(SF.black(100.0, K, 0.6**2 * 0.1, cp))
        assert SF.implied_vol(p, 100.0, K, 0.1, cp) == pytest.approx(0.6, abs=1e-6)
    assert np.isnan(
        SF.implied_vol(0.0, 100.0, 110.0, 0.1, "C")
    )  # a zero price is at or below intrinsic


def _surface(rho=-0.3, phi_scale=0.8):
    theta = np.array([0.0004, 0.001, 0.003])
    cap = np.minimum(4 / (theta * (1 + abs(rho))), 2 / np.sqrt(theta * (1 + abs(rho))))
    ex = [
        U("2026-10-21 08:00"),
        U("2026-10-22 08:00"),
        U("2026-10-27 08:00"),
    ]  # kept away from the dst change since the clock works in wall clock minutes
    asof = U("2026-10-20 08:00")
    T = np.array([(e - asof).total_seconds() / SF.YEAR_S for e in ex])
    return SF.Surface(
        "BTC",
        asof,
        "slices",
        ex,
        T,
        np.full(3, 100.0),
        theta,
        np.full(3, rho),
        cap * phi_scale,
        0.0,
        1.0,
        0,
        {},
    )


def test_digital_is_minus_dC_dK():
    s = _surface()
    target = U("2026-10-21 20:00")
    clock = SF.VarianceClock.flat()
    K = np.array([95.0, 100.0, 104.0])
    d = SF.digitals(s, target, K, clock)
    h = 1e-3

    def C(k_):
        w, _, F = SF.w_at(s, target, np.log(k_ / 100.0), clock)
        return SF.black(F, k_, w, "C")

    fd = -(C(K + h) - C(K - h)) / (2 * h)
    np.testing.assert_allclose(d["p"], fd, atol=1e-5)
    assert (d["p"] > d["naive"]).iloc[
        1
    ]  # negative skew puts the atm digital above N(d2)


def test_slices_butterfly_free_and_interpolation():
    s = _surface(phi_scale=0.99)
    assert all(SF.butterfly_g(s, i).min() >= -1e-9 for i in range(3))
    assert (SF.calendar_gaps(s) > 0).all()
    clock = SF.VarianceClock.flat()
    w_mid, _, _ = SF.w_at(
        s, U("2026-10-21 20:00"), [0.0], clock
    )  # half way between slices 0 and 1 on a flat clock
    assert w_mid[0] == pytest.approx((s.theta[0] + s.theta[1]) / 2, rel=1e-9)
    w_pre, _, _ = SF.w_at(
        s, U("2026-10-20 20:00"), [0.0], clock
    )  # half way to the first expiry
    assert w_pre[0] == pytest.approx(s.theta[0] / 2, rel=1e-9)


def test_fit_recovers_a_synthetic_surface():
    true = _surface(rho=-0.25, phi_scale=0.6)
    rows = []
    for i, e in enumerate(true.expiries):
        for K in np.linspace(85, 115, 25):
            w = float(true.w(np.log(K / 100.0), i))
            cp = "C" if K >= 100 else "P"
            p = float(SF.black(100.0, K, w, cp)) / 100.0  # priced in coin like deribit
            iv = np.sqrt(w / true.T[i])
            rows.append(
                {
                    "expiry": e,
                    "strike": K,
                    "cp": cp,
                    "underlying_price": 100.0,
                    "bid": p * 0.99,
                    "ask": p * 1.01,
                    "mark_price": p,
                    "mark_iv": 100 * iv,
                }
            )
    d = SF.prepare(pd.DataFrame(rows), true.asof)
    s = SF.fit_ssvi(d, "BTC", true.asof)
    np.testing.assert_allclose(s.theta, true.theta, rtol=0.03)
    assert s.rmse_iv < 0.5 and s.in_spread > 0.9


def test_prepare_on_a_deribit_shaped_payload():
    fx = json.loads(
        (
            Path(__file__).parent / "fixtures" / "deribit_book_summary_synthetic.json"
        ).read_text()
    )
    ts = datetime(2026, 10, 30, 3, 10, tzinfo=timezone.utc)
    raw = pd.DataFrame(
        [
            SF.parse_deribit_option(x, fx["index_price"]["result"]["index_price"], ts)
            for x in fx["book_summary"]["result"]
        ]
    )
    d = SF.prepare(raw, ts)
    otm = ((d["cp"] == "C") & (d["strike"] >= d["F"])) | (
        (d["cp"] == "P") & (d["strike"] < d["F"])
    )  # deep itm inverts poorly
    ok = d["iv_mark_price"].notna() & otm
    assert ok.any()
    np.testing.assert_allclose(
        100 * d.loc[ok, "iv_mark_price"], d.loc[ok, "mark_iv"], atol=3.0
    )  # coin price times forward reproduces the mark iv
