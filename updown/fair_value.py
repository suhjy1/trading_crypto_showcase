"""
Fair value of short-dated crypto binaries, priced as European and Asian (TWAP) digitals.

P(up) = F((ln(E/K) - V/2) / sqrt(V))

V is the variance to settlement, which I forecast with a seasonal EWMA. A mean-reverting variance level on deseasonalized 1-minute returns is multiplied by an intraday profile (weekday/weekend x ET hour), then integrated exactly over fractional minutes in forecast_var and season_sum. I also keep a flat EWMA, 24-hour realized variance and HAR as comparison models.

For TWAP settlement, E and V treat the prints already inside the averaging window as known. The remaining variance then scales with the cube of the window share still to come (see twap_moments).

F sets the tails. It is Gaussian by default, or the empirical CDF of past standardized moves for each time-to-expiry bucket, stored at 2,001 quantiles and refit on past data only (fit_cdfs and fair_value_cal).

Every feature at time t only uses candles that closed by t. test_no_look_ahead in tests/test_fair_value.py checks this.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

from updown.spec import ET, Spec

# binance spot price ticks
SPOT_TICK = {
    "BTCUSDT": 0.01,
    "ETHUSDT": 0.01,
    "SOLUSDT": 0.01,
    "XRPUSDT": 0.0001,
}
WEEK_MIN = 7 * 1440
SEASON_INTEGRAL_VERSION = "fractional_minutes_v1"
METHODS = ("rv24", "ewma", "season", "har")
HAR_W = (5, 30, 240, 1440)  # minutes of past r^2 in the har regression


def vol_features(
    bars: pd.DataFrame,
    *,
    hl_short: float = 60,
    long_days: int = 14,
    season_days: int = 28,
    clip_sd: float = 5.0,
) -> pd.DataFrame:
    c = bars["close"].astype(float)
    t = bars.index + timedelta(minutes=1)
    r = np.log(c).diff()
    cap = clip_sd * 1.4826 * r.abs().rolling(1440, min_periods=720).median()
    r2 = r.clip(-cap, cap).pow(2).to_numpy()
    f = pd.DataFrame(
        {
            "S": c.to_numpy(),
            "high": bars["high"].to_numpy(float),
            "low": bars["low"].to_numpy(float),
            "r2": r2,
        },
        index=t,
    )
    f["rv24"] = (
        r.pow(2).rolling(1440, min_periods=720).mean().to_numpy()
    )  
    f["lv_raw_s"] = (
        f["r2"].ewm(halflife=hl_short, min_periods=60, ignore_na=True).mean()
    )
    f["lv_raw_l"] = f["r2"].rolling(long_days * 1440, min_periods=1440).mean()
    et = bars.index.tz_convert(ET)
    bucket = (et.dayofweek >= 5).astype(int) * 24 + et.hour
    day = pd.Index(et.normalize().tz_localize(None))
    prof = _profiles(r.abs().to_numpy(), bucket, day, season_days)
    s = np.array([prof[d][b] if d in prof else np.nan for d, b in zip(day, bucket)])
    f["season"] = s
    z2 = f["r2"] / f["season"]
    f["lv_s"] = z2.ewm(halflife=hl_short, min_periods=60, ignore_na=True).mean()
    f["lv_l"] = z2.rolling(long_days * 1440, min_periods=1440).mean()
    for w in HAR_W:
        f[f"har_{w}"] = f["r2"].rolling(w, min_periods=max(1, w // 2)).mean()
    f["bucket"] = np.asarray(bucket)
    f["day"] = day  # profile date used for future minutes
    f.attrs["profiles"] = prof
    return f


_BUCKET_WEIGHT = np.r_[np.full(24, 5 * 60), np.full(24, 2 * 60)]


def _profiles(
    abs_r: np.ndarray, bucket: np.ndarray, day: pd.Index, season_days: int
) -> dict:
    df = pd.DataFrame({"a": abs_r, "b": bucket, "d": day}).dropna()
    days = pd.Index(sorted(df["d"].unique()))
    by_day = {d: g for d, g in df.groupby("d")}
    out = {}
    for i in range(season_days, len(days)):
        w = pd.concat([by_day[d] for d in days[i - season_days : i]])
        med = w.groupby("b")["a"].median().reindex(range(48)).to_numpy() ** 2
        med = np.where(np.isfinite(med) & (med > 0), med, np.nan)
        if np.isnan(med).all():
            continue
        med = np.where(np.isnan(med), np.nanmean(med), med)
        out[days[i]] = med / (np.sum(med * _BUCKET_WEIGHT) / _BUCKET_WEIGHT.sum())
    return out


def _week_cumsum(prof: np.ndarray) -> np.ndarray:
    m = np.arange(WEEK_MIN)
    dow, hour = m // 1440, (m % 1440) // 60
    s = prof[(dow >= 5).astype(int) * 24 + hour]
    return np.concatenate([[0.0], np.cumsum(s)])


def _et_week_minutes(ts: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    loc = ts.tz_convert(ET).tz_localize(None)
    elapsed = loc - pd.Timestamp("2024-01-01")
    minute = pd.Timedelta(minutes=1)
    return (elapsed // minute).to_numpy(), ((elapsed % minute) / minute).to_numpy()


def season_sum(
    prof: np.ndarray, a: pd.DatetimeIndex, b: pd.DatetimeIndex
) -> np.ndarray:
    cw = _week_cumsum(prof)
    ma, fa = _et_week_minutes(a)
    mb, fb = _et_week_minutes(b)
    wa, ia = np.divmod(ma, WEEK_MIN)
    wb, ib = np.divmod(mb, WEEK_MIN)
    whole = (wb - wa) * cw[-1] + (cw[ib] - cw[ia])
    partial = fb * (cw[ib + 1] - cw[ib]) - fa * (cw[ia + 1] - cw[ia])
    return np.maximum(whole + partial, 0.0)


def forecast_var(
    feat: pd.DataFrame,
    t: pd.DatetimeIndex,
    T,
    method: str = "season",
    kappa: float = 360,
) -> np.ndarray:
    x = feat.loc[t]
    T = pd.DatetimeIndex(
        np.broadcast_to(
            pd.DatetimeIndex([T]) if isinstance(T, pd.Timestamp) else T, len(t)
        )
    )
    n = np.maximum((T - t) / pd.Timedelta(minutes=1), 0.0).to_numpy(float)
    if method == "rv24":
        return x["rv24"].to_numpy() * n
    shrink = np.where(
        n > 0, kappa * (1 - np.exp(-n / kappa)) / np.maximum(n, 1e-9), 1.0
    )
    if method == "ewma":
        L, Sh = x["lv_raw_l"].to_numpy(), x["lv_raw_s"].to_numpy()
        return (L + (Sh - L) * shrink) * n
    if method == "har":
        return har_rate(x, np.maximum(n, 1.0), feat.attrs["har"]) * n
    if method == "season":
        L, Sh = x["lv_l"].to_numpy(), x["lv_s"].to_numpy()
        prof = feat.attrs["profiles"]
        out = np.full(len(t), np.nan)
        days = x["day"].to_numpy()
        for d in pd.unique(days):
            if d not in prof:
                continue
            i = days == d
            out[i] = season_sum(prof[d], t[i], T[i])
        return (L + (Sh - L) * shrink) * out
    raise ValueError(method)


def _har_x(x: pd.DataFrame, h: np.ndarray) -> np.ndarray:
    logs = [np.log(x[f"har_{w}"].to_numpy(float) + 1e-14) for w in HAR_W]
    return np.column_stack([*logs, np.log(h), np.eye(48)[x["bucket"].to_numpy(int)]])


def har_rate(x: pd.DataFrame, h: np.ndarray, har: dict) -> np.ndarray:
    return (
        np.exp(_har_x(x, h) @ np.asarray(har["coef"]) + har["intercept"]) * har["smear"]
    )


def fit_har(
    feats: dict[str, pd.DataFrame],
    until: pd.Timestamp,
    *,
    every: int = 7,
    horizons=(1, 2, 5, 15, 60, 240, 1440),
) -> dict:
    X, y = [], []
    for f in feats.values():
        r2 = f["r2"].fillna(0.0).to_numpy()
        cs = np.concatenate([[0.0], np.cumsum(r2)])
        ok = f[[f"har_{w}" for w in HAR_W]].notna().all(axis=1).to_numpy() & (
            f.index < until
        )
        pos = np.flatnonzero(ok)[::every]
        for h in horizons:
            p = pos[pos + h < len(f)]
            p = p[f.index[p + h] < until]
            if len(p) == 0:
                continue
            # mean r^2 of the candles closing after t up to t + h
            target = (cs[p + h + 1] - cs[p + 1]) / h
            X.append(_har_x(f.iloc[p], np.full(len(p), float(h))))
            y.append(np.log(target + 1e-14))
    X, y = np.vstack(X), np.concatenate(y)
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    fit = np.exp(A @ beta)
    return {
        "intercept": float(beta[0]),
        "coef": beta[1:].tolist(),
        "smear": float(np.mean(np.exp(y)) / np.mean(fit)),
        "n": int(len(y)),
    }


def twap_moments(
    feat: pd.DataFrame,
    t: pd.DatetimeIndex,
    T: pd.Timestamp,
    w_s: int,
    S: np.ndarray,
    method: str,
    known: tuple | None = None,
    discrete: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    W = pd.Timedelta(seconds=w_s)
    v_all = forecast_var(feat, t, T, method)
    v_pre = forecast_var(feat, t, T - W, method)
    S = np.asarray(S, float)
    if known is None:
        E = S
    else:
        s_known, n_known = (np.asarray(a, float) for a in known)
        E = (s_known + (w_s - n_known) * S) / w_s
    dv = np.maximum(v_all - v_pre, 0.0)
    if not discrete:
        share = np.clip(np.asarray((T - t) / W, float), 0.0, 1.0)
        return E, v_pre + dv * share**2 / 3
    w = float(w_s)
    left = np.maximum(np.asarray((T - t) / pd.Timedelta(seconds=1), float), 0.0)
    span = np.minimum(left, w)
    k = np.minimum(np.floor(left + 1e-9), w)
    r = np.where(span > 0, dv / np.maximum(span, 1e-12), 0.0)
    var_win = np.where(
        k < w,
        r * k * (k + 1) * (2 * k + 1) / (6 * w**2),
        r * (w + 1) * (2 * w + 1) / (6 * w),
    )
    return E, (v_pre + var_win) * (S / E) ** 2


def _cdf(cdf: Callable | None):
    return cdf or norm.cdf


def p_above(S, K, V, cdf: Callable | None = None) -> np.ndarray:
    S, K, V = (np.asarray(a, float) for a in (S, K, V))
    sd = np.sqrt(np.maximum(V, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (np.log(S / K) - V / 2) / sd
    z = np.where(sd > 0, z, np.where(S > K, np.inf, -np.inf))
    return _cdf(cdf)(z)


def p_touch(S, K, V, ext, up: bool, cdf: Callable | None = None) -> np.ndarray:
    S, K, V, ext = (np.asarray(a, float) for a in (S, K, V, ext))
    hit = ext >= K if up else ext <= K
    sd = np.sqrt(np.maximum(V, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(np.log(K / S)) / sd
    p = np.clip(2 * (1 - _cdf(cdf)(np.where(sd > 0, z, np.inf))), 0, 1)
    return np.where(hit, 1.0, p)


def ref_instant(spec: Spec, instant: pd.Timestamp) -> pd.Timestamp:
    return instant + timedelta(minutes=spec.ref_offset + 1)


def fair_value(
    spec: Spec,
    feat: pd.DataFrame,
    t: pd.DatetimeIndex,
    *,
    method: str = "season",
    cdf: Callable | None = None,
    basis_var: float = 0.0,
    base: float | None = None,
    twap_known: tuple | None = None,
    tie_tick: float = 0.0,
    twap_discrete: bool = False,
) -> np.ndarray:
    x = feat.loc[t]
    S = x["S"].to_numpy()
    T = ref_instant(spec, spec.end)
    V = forecast_var(feat, t, T, method)
    if spec.kind == "above":
        return p_above(S, spec.lo, V, cdf)
    if spec.kind == "below":
        return 1 - p_above(S, spec.lo, V, cdf)
    if spec.kind == "between":
        return p_above(S, spec.lo, V, cdf) - p_above(S, spec.hi, V, cdf)
    if spec.kind in ("updown", "updown_cl"):
        b_inst = (
            ref_instant(spec, spec.start)
            if spec.end - spec.start == timedelta(days=1)
            else spec.start
        )
        base = feat["S"].get(b_inst, np.nan) if base is None else base
        started = t >= b_inst
        if spec.twap_s:
            S, V = twap_moments(
                feat, t, T, spec.twap_s, S, method, twap_known, discrete=twap_discrete
            )
        elif tie_tick > 0:
            base = base - tie_tick / 2
        if spec.kind == "updown_cl":
            V = V + basis_var
        return np.where(started, p_above(S, base, V, cdf), 0.5)
    if spec.kind in ("touch_up", "touch_down"):
        up = spec.kind == "touch_up"
        col = "high" if up else "low"
        # only candles opening at or after the window start
        w = feat[col].where(feat.index > spec.start)
        ext = (w.cummax() if up else w.cummin()).reindex(t).to_numpy()
        ext = np.where(np.isnan(ext), -np.inf if up else np.inf, ext)
        return p_touch(S, spec.lo, V, ext, up, cdf)
    raise ValueError(spec.kind)


TAU_BUCKETS = (0, 30, 360, 2880, 1e9)  # minutes to the reference instant
CAL_HORIZONS = (5, 15, 60, 240, 1440, 4320, 10080)
N_QUANTILES = 2001


def fit_cdfs(
    feats: dict[str, pd.DataFrame],
    until: pd.Timestamp,
    *,
    every: int = 7,
    method: str = "season",
) -> dict[int, np.ndarray]:
    zs: dict[int, list] = {b: [] for b in range(len(TAU_BUCKETS) - 1)}
    for f in feats.values():
        idx = f.index[(f.index < until) & f["lv_l"].notna() & f["season"].notna()][
            ::every
        ]
        for h in CAL_HORIZONS:
            T = idx + timedelta(minutes=h)
            ok = T.isin(f.index) & (T < until)
            t, T = idx[ok], T[ok]
            if len(t) == 0:
                continue
            V = forecast_var(f, t, T, method)
            z = (
                np.log(f["S"].reindex(T).to_numpy() / f["S"].reindex(t).to_numpy())
                + V / 2
            ) / np.sqrt(V)
            zs[_bucket(h)].append(z[np.isfinite(z)])
    out = {}
    for b, parts in zs.items():
        z = np.concatenate(parts) if parts else np.array([])
        if len(z) > 1000:
            out[b] = np.quantile(z, np.linspace(0, 1, N_QUANTILES))
    return out


def _bucket(tau_min) -> np.ndarray | int:
    return np.searchsorted(TAU_BUCKETS, tau_min, side="left") - 1


def cdf_from_quantiles(q: np.ndarray) -> Callable:
    p = np.linspace(0, 1, len(q))
    return lambda x: np.interp(x, q, p, left=0.0, right=1.0)


def fair_value_cal(
    spec: Spec,
    feat: pd.DataFrame,
    t: pd.DatetimeIndex,
    cdfs: dict[int, np.ndarray],
    *,
    method: str = "season",
    basis_var: float = 0.0,
    base: float | None = None,
    twap_known: tuple | None = None,
    tie_tick: float = 0.0,
    twap_discrete: bool = False,
) -> np.ndarray:
    tau = ((ref_instant(spec, spec.end) - t) / pd.Timedelta(minutes=1)).to_numpy(float)
    b = _bucket(tau)
    out = np.full(len(t), np.nan)
    for k in np.unique(b):
        i = b == k
        cdf = cdf_from_quantiles(cdfs[k]) if k in cdfs else None
        kn = (
            None
            if twap_known is None
            else tuple(
                np.broadcast_to(np.asarray(a, float), len(t))[i] for a in twap_known
            )
        )
        out[i] = fair_value(
            spec,
            feat,
            t[i],
            method=method,
            cdf=cdf,
            basis_var=basis_var,
            base=base,
            twap_known=kn,
            tie_tick=tie_tick,
            twap_discrete=twap_discrete,
        )
    return out
