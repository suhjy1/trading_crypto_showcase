from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from updown import fair_value as FV
from updown.spec import parse_spec, settle

U = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
CREATION = (
    'This market will resolve to "Yes" if any Binance 1 minute candle for BTC/USDT from the creation of this market '
    "through 11:59 PM ET on the last day of the date range specified in the title has a final High price ..."
)


@pytest.mark.parametrize(
    "title,close,open_,rules,exp",
    [
        (
            "Bitcoin Up or Down - October 24, 2:30AM-2:45AM ET",
            "2026-10-24 06:45",
            None,
            None,
            ("updown_cl", "BTCUSDT", "2026-10-24 06:30", None, -1),
        ),
        (
            "XRP Up or Down - October 29, 10:05AM-10:10AM ET",
            "2026-10-29 14:10",
            None,
            None,
            ("updown_cl", "XRPUSDT", "2026-10-29 14:05", None, -1),
        ),
        (
            "Bitcoin Up or Down - October 23, 8AM ET",
            "2026-10-23 13:00",
            None,
            None,
            ("updown", "BTCUSDT", "2026-10-23 12:00", None, -1),
        ),
        (
            "Bitcoin Up or Down on October 29?",
            "2026-10-29 16:00",
            None,
            None,
            ("updown", "BTCUSDT", "2026-10-28 16:00", None, 0),
        ),
        (
            "Ethereum above 2,450 on October 24, 4AM ET?",
            "2026-10-24 08:00",
            None,
            None,
            ("above", "ETHUSDT", None, 2450, -1),
        ),
        (
            "Will the price of Ethereum be above $1,900 on October 29?",
            "2026-10-29 16:00",
            None,
            None,
            ("above", "ETHUSDT", None, 1900, -1),
        ),
        (
            "Will the price of Bitcoin be less than $76,000 on September 28?",
            "2026-09-28 16:00",
            None,
            None,
            ("below", "BTCUSDT", None, 76000, -1),
        ),
        (
            "Will the price of Bitcoin be between $74,000 and $76,000 on October 29?",
            "2026-10-29 16:00",
            None,
            None,
            ("between", "BTCUSDT", None, 74000, -1),
        ),
        (
            "Will Ethereum reach $2,400 on August 21?",
            "2026-08-22 04:00",
            None,
            None,
            ("touch_up", "ETHUSDT", "2026-08-21 04:00", 2400, -1),
        ),
        (
            "Will Bitcoin dip to $77,000 on September 20?",
            "2026-09-21 04:00",
            None,
            None,
            ("touch_down", "BTCUSDT", "2026-09-20 04:00", 77000, -1),
        ),
        (
            "Will Ethereum reach $2,300 August 17-23?",
            "2026-08-24 04:00",
            None,
            None,
            ("touch_up", "ETHUSDT", "2026-08-17 04:00", 2300, -1),
        ),
        (
            "Will Bitcoin dip to $78,000 August 31-September 6?",
            "2026-09-07 04:00",
            None,
            None,
            ("touch_down", "BTCUSDT", "2026-08-31 04:00", 78000, -1),
        ),
        (
            "Will Bitcoin reach $105,000 in October?",
            "2026-11-01 04:00",
            None,
            None,
            ("touch_up", "BTCUSDT", "2026-10-01 04:00", 105000, -1),
        ),
        (
            "Will Bitcoin reach $86,000 October 5-11?",
            "2026-10-12 04:00",
            "2026-10-06 16:57:07",
            CREATION,
            ("touch_up", "BTCUSDT", "2026-10-06 16:57", 86000, -1),
        ),
    ],
)
def test_parse_spec(title, close, open_, rules, exp):
    s = parse_spec(title, U(close), U(open_) if open_ else None, rules)
    kind, sym, start, lo, off = exp
    assert (s.kind, s.symbol, s.ref_offset) == (kind, sym, off)
    assert s.end == U(close)
    assert s.start == (U(start) if start else None)
    assert lo is None or s.lo == lo


def test_parse_rejects_mismatch_and_other_templates():
    assert (
        parse_spec(
            "Bitcoin Up or Down - October 24, 2:30AM-2:45AM ET", U("2026-10-24 07:00")
        )
        is None
    )  # close_ts doesn't match the title
    assert (
        parse_spec(
            "Will the price of Bitcoin be above $80,000 on October 29?",
            U("2026-10-29 17:00"),
        )
        is None
    )  # not noon ET
    assert (
        parse_spec(
            "Will Bitcoin have 4 green days the week of October 19?",
            U("2026-10-26 04:00"),
        )
        is None
    )
    assert parse_spec("Bitcoin ETF Flows on October 21?", U("2026-10-22 04:00")) is None
    s = parse_spec(
        "Will the price of Bitcoin be between $76,000 and $74,000 on October 29?",
        U("2026-10-29 16:00"),
    )
    assert (s.lo, s.hi) == (74000, 76000)


def _bars(start: str, closes) -> pd.DataFrame:
    idx = pd.date_range(U(start), periods=len(closes), freq="1min")
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame(
        {
            "open": o,
            "high": np.maximum(o, c) + 0.5,
            "low": np.minimum(o, c) - 0.5,
            "close": c,
            "volume": 1.0,
        },
        index=idx,
    )


def test_settle_reference_candles():
    b = _bars(
        "2026-10-29 15:55", [100, 101, 102, 103, 104, 110, 120, 130]
    )  # 15:59 opens at 104, 16:00 at 110
    above = parse_spec(
        "Will the price of Bitcoin be above $105 on October 29?", U("2026-10-29 16:00")
    )
    assert (
        settle(above, b) == 0.0
    )  # the candle that opened 15:59 and closes at 16:00 closed at 104
    assert (
        settle(
            parse_spec(
                "Will the price of Bitcoin be above $103 on October 29?",
                U("2026-10-29 16:00"),
            ),
            b,
        )
        == 1.0
    )
    between = parse_spec(
        "Will the price of Bitcoin be between $104 and $106 on October 29?",
        U("2026-10-29 16:00"),
    )
    assert settle(between, b) == 1.0  # lower bound inclusive


def test_settle_touch_and_creation_window():
    b = _bars("2026-10-05 04:00", [100.0] * 60 + [120.0] + [100.0] * 60)
    s = parse_spec(
        "Will Bitcoin reach $110 October 5-11?", U("2026-10-05 06:01"), None, None
    )
    assert s is None  # weekly window end has to be midnight ET
    s = parse_spec("Will Bitcoin reach $110 on October 5?", U("2026-10-06 04:00"))
    w = _bars("2026-10-05 04:00", [100.0] * 1440)
    assert settle(s, w) == 0.0
    w.iloc[700, w.columns.get_loc("high")] = 111
    assert settle(s, w) == 1.0


def test_pricing_identities():
    S, V = 100.0, 0.0004
    a, b, c = (FV.p_above(S, k, V) for k in (95, 100, 105))
    assert a > b > c and abs(b - 0.5) < 0.01
    assert FV.p_above(S, 99, 0.0) == 1.0 and FV.p_above(S, 101, 0.0) == 0.0
    # brackets partition the line
    ks = [0.01, 90, 95, 100, 105, 110, 1e9]
    assert (
        abs(
            sum(
                FV.p_above(S, lo, V) - FV.p_above(S, hi, V)
                for lo, hi in zip(ks, ks[1:])
            )
            - 1
        )
        < 1e-9
    )
    k = np.linspace(1e-6, 400, 400_001)
    assert abs(np.trapezoid(FV.p_above(S, k, 0.04), k) - S) < 0.05
    # touch is at least the terminal probability and goes to 1 once crossed
    assert FV.p_touch(S, 105, V, ext=101, up=True) > FV.p_above(S, 105, V)
    assert FV.p_touch(S, 105, V, ext=106, up=True) == 1.0


def _random_bars(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    return _bars("2026-07-01 04:00", c)


def test_season_sum_matches_bruteforce():
    prof = np.linspace(0.5, 1.5, 48)
    a = pd.DatetimeIndex(
        [U("2026-10-30 21:17"), U("2026-11-01 05:00")]
    )  # spans a weekend and the dst change
    b = pd.DatetimeIndex([U("2026-11-03 08:41"), U("2026-11-01 07:30")])
    got = FV.season_sum(prof, a, b)
    for i in range(2):
        mins = pd.date_range(a[i], b[i], freq="1min", inclusive="left").tz_convert(
            FV.ET
        )
        exp = prof[(mins.dayofweek >= 5).astype(int) * 24 + mins.hour].sum()
        assert (
            abs(got[i] - exp) <= 60 * prof.max()
        )  # the dst hour can count one time more or less


@pytest.mark.parametrize(
    "start,end,expected",
    [
        (
            "2026-10-30 17:59:15",
            "2026-10-30 17:59:45",
            1.0,
        ),  # 30 seconds at the 13:00 ET rate of 2
        ("2026-10-30 17:59:59.750", "2026-10-30 18:00", 2 / 240),
        (
            "2026-10-30 17:59:45",
            "2026-10-30 18:00:15",
            1.5,
        ),  # 15 seconds each at rates 2 and 4
        ("2026-10-30 17:59", "2026-10-30 18:01", 6.0),  # whole minutes
        ("2026-10-30 18:00:15", "2026-10-30 18:00:15", 0.0),
        ("2026-10-30 18:00:15", "2026-10-30 17:59:45", 0.0),  # expired horizon
        (
            "2026-10-31 03:59:30",
            "2026-10-31 04:00:30",
            6.0,
        ),  # weekday into weekend at rates 5 and 7
        (
            "2026-10-26 03:59:30",
            "2026-10-26 04:00:30",
            10.0,
        ),  # sunday into monday at rates 9 and 11
        (
            "2024-01-01 04:59:30",
            "2024-01-01 05:00:30",
            10.0,
        ),  # same boundary before the helper's epoch
    ],
)
def test_season_sum_fractional_boundaries(start, end, expected):
    prof = np.ones(48)
    prof[[13, 14, 23, 24, 47, 0]] = [2, 4, 5, 7, 9, 11]
    got = FV.season_sum(prof, pd.DatetimeIndex([U(start)]), pd.DatetimeIndex([U(end)]))
    np.testing.assert_allclose(got, [expected], rtol=1e-12, atol=1e-12)


def test_season_variance_matches_flat_ewma_at_fractional_times():
    end = U("2026-10-30 18:00")
    seconds = np.array([120, 60, 45, 30, 15, 5, 1, 0.25, 0, -1])
    t = pd.DatetimeIndex(end - pd.to_timedelta(seconds, unit="s"))
    day = pd.Timestamp("2026-10-30")
    f = pd.DataFrame(
        {"lv_s": 4e-8, "lv_l": 1e-8, "lv_raw_s": 4e-8, "lv_raw_l": 1e-8, "day": day},
        index=t,
    )
    f.attrs["profiles"] = {day: np.ones(48)}
    season = FV.forecast_var(f, t, end, "season")
    ewma = FV.forecast_var(f, t, end, "ewma")
    np.testing.assert_allclose(season, ewma, rtol=1e-10, atol=1e-20)
    assert np.all(season[-2:] == 0)


@pytest.mark.parametrize("window_s", [30, 60])
def test_season_twap_variance_has_cubic_remaining_time(window_s):
    end = U("2026-10-30 18:00")
    seconds = np.array([window_s, window_s / 2, window_s / 4, 5, 1, 0.25, 0])
    t = pd.DatetimeIndex(end - pd.to_timedelta(seconds, unit="s"))
    day = pd.Timestamp("2026-10-30")
    level = 1e-8  # variance per minute
    f = pd.DataFrame({"lv_s": level, "lv_l": level, "day": day}, index=t)
    f.attrs["profiles"] = {day: np.ones(48)}
    known_n = window_s - seconds
    expected_end, variance = FV.twap_moments(
        f,
        t,
        end,
        window_s,
        np.full(len(t), 100.0),
        "season",
        known=(100.0 * known_n, known_n),
    )
    expected_var = level * (seconds / 60) * (seconds / window_s) ** 2 / 3
    np.testing.assert_allclose(variance, expected_var, rtol=1e-10, atol=1e-22)
    np.testing.assert_allclose(expected_end, 100.0)
    assert variance[-1] == 0


@pytest.mark.parametrize("method", FV.METHODS)
def test_no_look_ahead(method):
    # fair values at t must not change when bars after t change
    bars = _random_bars(40 * 1440)
    spec = parse_spec(
        "Will the price of Bitcoin be above $100 on August 9?", U("2026-08-09 16:00")
    )
    t = pd.DatetimeIndex([U("2026-08-08 10:00"), U("2026-08-09 15:00")])
    cut = t.max()

    def feat(b):
        f = FV.vol_features(b)
        if (
            method == "har"
        ):  # fit on windows that ended before the first decision instant
            f.attrs["har"] = FV.fit_har({"x": f}, t.min())
        return f

    p0 = FV.fair_value(spec, feat(bars), t, method=method)
    alt = bars.copy()
    later = alt.index >= cut  # candles opening at or after the last decision instant
    alt.loc[later, ["open", "high", "low", "close"]] *= np.exp(
        np.random.default_rng(1).normal(0, 0.05, (later.sum(), 1))
    )
    p1 = FV.fair_value(spec, feat(alt), t, method=method)
    assert np.isfinite(p0).all()
    np.testing.assert_allclose(p0, p1)


def test_touch_uses_window_extreme_only():
    bars = _random_bars(40 * 1440)
    spec = parse_spec("Will Bitcoin reach $1,000 on August 9?", U("2026-08-10 04:00"))
    f = FV.vol_features(bars)
    t = pd.DatetimeIndex([U("2026-08-09 12:00")])
    f.loc[U("2026-08-09 02:00"), "high"] = 2000  # before the window so it's ignored
    assert FV.fair_value(spec, f, t)[0] < 0.01
    f.loc[U("2026-08-09 05:00"), "high"] = 2000  # inside the window so it counts as touched
    assert FV.fair_value(spec, f, t)[0] == 1.0






def test_chainlink_updown_is_twap():
    s = parse_spec(
        "Bitcoin Up or Down - October 24, 2:30AM-2:45AM ET", U("2026-10-24 06:45")
    )
    assert s.twap_s == 60
    assert (
        parse_spec(
            "Bitcoin Up or Down - October 24, 2:30AM-2:45AM ET",
            U("2026-10-24 06:45"),
            rules="Chainlink BTC/USD price",
        ).twap_s
        == 0
    )
    assert (
        parse_spec(
            "Bitcoin Up or Down - October 23, 8AM ET", U("2026-10-23 13:00")
        ).twap_s
        == 0
    )  # binance hourly settles on a point price
    r30 = "Chainlink BTC/USD TWAP data stream at https://data.chain.link/streams/btc-usd-twap-30s-streams."
    assert (
        parse_spec(
            "Bitcoin Up or Down - August 10, 2:30AM-2:35AM ET",
            U("2026-08-10 06:35"),
            rules=r30,
        ).twap_s
        == 30
    )


def test_twap_moments():
    f = FV.vol_features(_random_bars(40 * 1440))
    f.attrs["har"] = FV.fit_har({"x": f}, U("2026-08-05"))
    T = U("2026-08-09 15:00")
    S = np.array([100.0])
    t = pd.DatetimeIndex([T - timedelta(minutes=10)])
    E, V = FV.twap_moments(f, t, T, 60, S, "rv24")
    v_all, v_pre = FV.forecast_var(f, t, T, "rv24"), FV.forecast_var(
        f, t, T - timedelta(seconds=60), "rv24"
    )
    assert E[0] == 100.0 and V[0] == pytest.approx(v_pre[0] + (v_all[0] - v_pre[0]) / 3)
    t = pd.DatetimeIndex([T - timedelta(seconds=15)])
    f = f.reindex(f.index.union(t), method="ffill").pipe(
        lambda g: (setattr(g, "attrs", f.attrs), g)[1]
    )  # add a live row between candles
    E, V = FV.twap_moments(f, t, T, 60, S, "har", known=(45 * 101.0, 45))
    assert E[0] == pytest.approx((45 * 101 + 15 * 100) / 60)
    v_full = FV.forecast_var(f, t, T, "har")[0]
    assert V[0] == pytest.approx(v_full * (15 / 60) ** 2 / 3)
    # at the end everything is known so there's no variance left
    E, V = FV.twap_moments(
        f, pd.DatetimeIndex([T]), T, 60, S, "har", known=(60 * 99.0, 60)
    )
    assert E[0] == pytest.approx(99.0) and V[0] == 0.0


def test_twap_fair_value_is_decided_late():
    f = FV.vol_features(_random_bars(40 * 1440))
    spec = parse_spec(
        "Bitcoin Up or Down - August 9, 10:55AM-11:00AM ET", U("2026-08-09 15:00")
    )
    t = pd.DatetimeIndex([U("2026-08-09 15:00") - timedelta(seconds=5)])
    S = float(f["S"].asof(t[0]))
    g = f.reindex(f.index.union(t), method="ffill")
    g.attrs = f.attrs
    g.loc[t[0], "S"] = S
    base = S
    p_twap = FV.fair_value(
        spec, g, t, method="rv24", base=base, twap_known=(55 * S * 1.0002, 55)
    )[0]
    p_point = FV.fair_value(
        spec.__class__(**{**spec.__dict__, "twap_s": 0}), g, t, method="rv24", base=base
    )[0]
    assert p_twap > 0.95 and abs(p_point - 0.5) < 0.05
