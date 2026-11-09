"""
Arbitrage-free implied vol surface fitted to Deribit options with SSVI, and the digitals it implies.

We wanted to know whether the options market could price the daily "above $K" ladders on Polymarket and Kalshi better than a realized-vol model. The answer depends on skew. A digital is -dC/dK, so on a skewed surface

P(S_T > K) = N(d2) - phi(d2) * (dw/dk) / (2 sqrt(w))

and that second skew term is worth 4 to 7c at the money on daily BTC and ETH ladders, and more just below it.

parse_deribit_option and prepare: turn quotes into forward-based Black-76 implied vols at the bid, ask and mark. Only liquid OTM options inside |ln K/F| <= 0.35 go into the fit.

fit_ssvi: Gatheral-Jacquier SSVI, either with one rho and phi per expiry (slices) or a single power-law phi (global). Total ATM variance theta cannot decrease by construction, so there is no calendar arbitrage, and each slice's phi is capped at the butterfly-free bound. butterfly_g and calendar_gaps check both conditions on the fitted surface.

VarianceClock and w_at: since the 12:00 ET targets land between Deribit's 08:00 UTC expiries, we interpolate between expiries in variance time instead of clock time, using the intraday seasonal profile from fair_value.

digitals: skew-aware digital probabilities next to the naive N(d2).

We parked this work because Deribit's shortest expiry is daily. Stretching its smile to the 15-minute and hourly contracts is guesswork, and near expiry the realized-vol model is the better tool.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.stats import norm


YEAR_S = 365.0 * 86400
K_MAX = 0.35  # max |ln(K/F)| that goes into the fit
MIN_T_DAYS = 2 / 24
MAX_SPREAD_IV = 0.30  # options with a wider bid ask spread in vol are left out
DERIBIT_API = "https://www.deribit.com/api/v2/public"
_MON = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
_INSTRUMENT = re.compile(r"^([A-Z]+)-(\d{1,2})([A-Z]{3})(\d{2})-(\d+(?:d\d+)?)-([CP])$")


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def parse_deribit_option(x: dict, index_price: float, ts: datetime) -> dict | None:
    # one row of Deribit's get_book_summary_by_currency, prices are in coin units
    m = _INSTRUMENT.match(x.get("instrument_name") or "")
    if not m or m.group(3) not in _MON:
        return None
    cur, day, mon, yy, k, cp = m.groups()
    return {
        "ts": ts,
        "instrument": x["instrument_name"],
        "currency": cur,
        "expiry": datetime(2000 + int(yy), _MON[mon], int(day), 8, tzinfo=timezone.utc),
        "strike": float(k.replace("d", ".")),
        "cp": cp,
        "bid": _num(x.get("bid_price")),
        "ask": _num(x.get("ask_price")),
        "mark_price": _num(x.get("mark_price")),
        "mark_iv": _num(x.get("mark_iv")),
        "underlying_price": _num(x.get("underlying_price")),
        "index_price": _num(index_price),
        "open_interest": _num(x.get("open_interest")),
        "volume_usd": _num(x.get("volume_usd")),
    }


def black(F, K, w, cp):
    F, K, w = (np.asarray(a, float) for a in (F, K, w))
    sd = np.sqrt(np.maximum(w, 1e-16))
    d1 = (np.log(F / K) + w / 2) / sd
    d2 = d1 - sd
    call = F * norm.cdf(d1) - K * norm.cdf(d2)
    return np.where(np.asarray(cp) == "C", call, call - F + K)


def implied_vol(price, F, K, T, cp) -> float:
    if not (np.isfinite(price) and price > 0 and T > 0):
        return np.nan
    intrinsic = max(F - K, 0.0) if cp == "C" else max(K - F, 0.0)
    upper = F if cp == "C" else K
    if price <= intrinsic + 1e-12 or price >= upper:
        return np.nan
    try:
        return brentq(
            lambda s: float(black(F, K, s * s * T, cp)) - price, 1e-4, 20.0, xtol=1e-7
        )
    except ValueError:
        return np.nan


def prepare(rows: pd.DataFrame, now: datetime) -> pd.DataFrame:
    d = rows.copy()
    d["T"] = (
        pd.to_datetime(d["expiry"], utc=True) - pd.Timestamp(now)
    ).dt.total_seconds() / YEAR_S
    d = d[d["T"] > 0].copy()
    d["F"] = d["underlying_price"]
    d["k"] = np.log(d["strike"] / d["F"])
    for c in ("bid", "ask", "mark_price"):
        d[f"{c}_usd"] = d[c] * d["F"]
    for c in ("bid", "ask", "mark_price"):
        d[f"iv_{c}"] = [
            implied_vol(p, F, K, T, cp)
            for p, F, K, T, cp in zip(
                d[f"{c}_usd"], d["F"], d["strike"], d["T"], d["cp"]
            )
        ]
    d["iv_mid"] = (d["iv_bid"] + d["iv_ask"]) / 2
    otm = ((d["cp"] == "C") & (d["strike"] >= d["F"])) | (
        (d["cp"] == "P") & (d["strike"] < d["F"])
    )
    spread = d["iv_ask"] - d["iv_bid"]
    d["fit"] = (
        otm
        & d["iv_mid"].notna()
        & (spread > 0)
        & (spread < MAX_SPREAD_IV)
        & (d["k"].abs() <= K_MAX)
        & (d["T"] >= MIN_T_DAYS / 365)
    )
    d["w_mid"] = d["iv_mid"] ** 2 * d["T"]
    d["weight"] = 1.0 / np.clip(spread, 0.005, None)
    return d


def _phi(theta, eta, gamma):
    return eta / (theta**gamma * (1 + theta) ** (1 - gamma))


def slice_w(k, theta, rho, phi):
    k = np.asarray(k, float)
    return theta / 2 * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2))


def slice_dw(k, theta, rho, phi):
    # dw/dk of one slice
    k = np.asarray(k, float)
    return (
        theta
        / 2
        * phi
        * (rho + (phi * k + rho) / np.sqrt((phi * k + rho) ** 2 + 1 - rho**2))
    )


@dataclass
class Surface:
    currency: str
    asof: pd.Timestamp
    kind: str  # global or slices
    expiries: list  # pd.Timestamp in ascending order
    T: np.ndarray  # years from asof
    F: np.ndarray  # forward per expiry
    theta: np.ndarray  # atm total variance per expiry, never decreasing
    rho: np.ndarray  # per expiry, all equal for global
    phi: np.ndarray  # per expiry
    rmse_iv: float  # weighted, in vol points
    in_spread: float
    n: int
    params: dict  # rho, eta and gamma for global
    spot: float = float("nan")

    def w(self, k, i: int):
        return slice_w(k, self.theta[i], self.rho[i], self.phi[i])

    def dw(self, k, i: int):
        return slice_dw(k, self.theta[i], self.rho[i], self.phi[i])


def _sig(x):
    return 1 / (1 + np.exp(-x))


def fit_ssvi(d: pd.DataFrame, currency: str, asof, kind: str = "slices") -> Surface:
    f = d[d["fit"]]
    ex = sorted(f["expiry"].unique())
    m = len(ex)
    T = np.array([f.loc[f["expiry"] == e, "T"].iloc[0] for e in ex])
    F = np.array([f.loc[f["expiry"] == e, "F"].iloc[0] for e in ex])
    ie = f["expiry"].map({e: i for i, e in enumerate(ex)}).to_numpy()
    k, Tr, wt = f["k"].to_numpy(), f["T"].to_numpy(), np.sqrt(f["weight"].to_numpy())
    iv_o = f["iv_mid"].to_numpy()
    th0 = []
    for e in ex:
        g = f[f["expiry"] == e].sort_values("k")
        th0.append(max(float(np.interp(0.0, g["k"], g["w_mid"])), 1e-7))
    inc0 = np.diff(np.r_[0.0, np.maximum.accumulate(th0)])

    def unpack(x):
        theta = np.cumsum(np.exp(x[:m]))  # can't decrease by construction
        if kind == "global":
            rho = np.full(m, np.tanh(x[m]))
            eta = 2 / (1 + abs(rho[0])) * _sig(x[m + 1])  # keeps eta * (1 + |rho|) < 2
            gamma = 0.5 * _sig(x[m + 2])  # between 0 and 1/2
            phi = eta / (theta**gamma * (1 + theta) ** (1 - gamma))
            return (
                theta,
                rho,
                phi,
                {"rho": float(rho[0]), "eta": float(eta), "gamma": float(gamma)},
            )
        rho = np.tanh(x[m : 2 * m])
        # cap on phi that keeps each slice butterfly free
        cap = np.minimum(4 / (theta * (1 + np.abs(rho))), 2 / np.sqrt(theta * (1 + np.abs(rho))))
        phi = cap * _sig(x[2 * m : 3 * m])
        return theta, rho, phi, {}

    def resid(x):
        theta, rho, phi, _ = unpack(x)
        return wt * (np.sqrt(slice_w(k, theta[ie], rho[ie], phi[ie]) / Tr) - iv_o)

    lt = np.log(np.maximum(inc0, 1e-9))
    x0 = (
        np.r_[lt, np.arctanh(-0.2), 0.0, 0.0]
        if kind == "global"
        else np.r_[lt, np.full(m, np.arctanh(-0.2)), np.zeros(m)]
    )
    sol = least_squares(resid, x0, method="trf")
    theta, rho, phi, params = unpack(sol.x)
    iv_m = np.sqrt(slice_w(k, theta[ie], rho[ie], phi[ie]) / Tr)
    rmse = float(np.sqrt(np.average((iv_m - iv_o) ** 2, weights=f["weight"])))
    inside = float(
        np.mean((iv_m >= f["iv_bid"].to_numpy()) & (iv_m <= f["iv_ask"].to_numpy()))
    )
    spot = (
        float(d["index_price"].iloc[0])
        if "index_price" in d and len(d)
        else float("nan")
    )
    return Surface(
        currency,
        pd.Timestamp(asof),
        kind,
        list(ex),
        T,
        F,
        theta,
        rho,
        phi,
        100 * rmse,
        inside,
        len(f),
        params,
        spot,
    )


def butterfly_g(s: Surface, i: int, k=np.linspace(-0.6, 0.6, 601)) -> np.ndarray:
    w, w1 = s.w(k, i), s.dw(k, i)
    w2 = np.gradient(w1, k)
    return (1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2


def calendar_gaps(s: Surface, k=np.linspace(-0.6, 0.6, 601)) -> np.ndarray:
    return np.array(
        [float(np.min(s.w(k, i + 1) - s.w(k, i))) for i in range(len(s.expiries) - 1)]
    )


class VarianceClock:

    def __init__(self, profile: np.ndarray) -> None:
        from updown.fair_value import season_sum

        self.profile, self._sum = np.asarray(profile, float), season_sum

    def between(self, a: pd.Timestamp, b: pd.Timestamp) -> float:
        return float(
            self._sum(self.profile, pd.DatetimeIndex([a]), pd.DatetimeIndex([b]))[0]
        )

    @classmethod
    def flat(cls) -> "VarianceClock":
        return cls(np.ones(48))


def _bracket(
    s: Surface, target: pd.Timestamp, clock: VarianceClock
) -> tuple[list[tuple[int, float]], float]:
    target = pd.Timestamp(target)
    ex = [pd.Timestamp(e) for e in s.expiries]
    j = int(np.searchsorted(np.array([e.value for e in ex]), target.value, side="left"))
    if j < len(ex) and ex[j] == target:
        return [(j, 1.0)], float(s.F[j])
    if j == 0:
        share = clock.between(s.asof, target) / max(clock.between(s.asof, ex[0]), 1e-12)
        tshare = (target - s.asof) / (ex[0] - s.asof)
        F = s.F[0] if not np.isfinite(s.spot) else s.spot * (s.F[0] / s.spot) ** tshare
        return [(0, share)], float(F)
    if j >= len(ex):
        raise ValueError("target after the last fitted expiry")
    share = clock.between(ex[j - 1], target) / max(
        clock.between(ex[j - 1], ex[j]), 1e-12
    )
    return [(j - 1, 1 - share), (j, share)], float(
        s.F[j - 1] + share * (s.F[j] - s.F[j - 1])
    )


def w_at(
    s: Surface, target: pd.Timestamp, k, clock: VarianceClock
) -> tuple[np.ndarray, np.ndarray, float]:
    # returns w(k), dw/dk and the forward
    mix, F = _bracket(s, target, clock)
    k = np.asarray(k, float)
    return sum(a * s.w(k, i) for i, a in mix), sum(a * s.dw(k, i) for i, a in mix), F


def digitals(s: Surface, target: pd.Timestamp, K, clock: VarianceClock) -> pd.DataFrame:
    K = np.asarray(K, float)
    _, _, F = w_at(s, target, [0.0], clock)
    k = np.log(K / F)
    w, dw, _ = w_at(s, target, k, clock)
    sd = np.sqrt(np.maximum(w, 1e-16))
    d2 = -k / sd - sd / 2
    T = max((pd.Timestamp(target) - s.asof).total_seconds() / YEAR_S, 1e-9)
    return pd.DataFrame(
        {
            "K": K,
            "k": k,
            "iv": np.sqrt(w / T),
            "p": norm.cdf(d2) - norm.pdf(d2) * dw / (2 * sd),
            "naive": norm.cdf(d2),
            "skew_term": -norm.pdf(d2) * dw / (2 * sd),
            "F": F,
        }
    )


def fetch_snapshot(currency: str) -> pd.DataFrame:
    import requests

    def get(path, params):
        return requests.get(DERIBIT_API + path, params=params, timeout=30).json()["result"]

    idx = get("/get_index_price", {"index_name": f"{currency.lower()}_usd"})["index_price"]
    res = get("/get_book_summary_by_currency", {"currency": currency, "kind": "option"})
    ts = datetime.now(timezone.utc)
    return pd.DataFrame([r for r in (parse_deribit_option(x, idx, ts) for x in res) if r])


def usdt_basis(binance_usdt: float, usd_index: float) -> float:
    return binance_usdt / usd_index


def reanchor(s: Surface, index_now: float) -> Surface:
    from dataclasses import replace

    r = index_now / s.spot if np.isfinite(s.spot) and s.spot > 0 else 1.0
    return replace(s, F=s.F * r, spot=float(index_now))
