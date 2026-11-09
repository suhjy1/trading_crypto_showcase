"""
Why the 15-minute contract is an Asian digital and not a European one.

A 15-minute up or down market settles on Chainlink's 60-second TWAP. Inside the final minute, part of the average is already known, so the contract gets decided faster than a point-price model expects.

I price one contract on synthetic data, with spot held 1 bp above the strike through the final 90 seconds, once as a TWAP digital and once as a point digital.

Run with: python examples/twap_vs_point.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing

from updown import fair_value as FV  # noqa: E402
from updown.spec import parse_spec  # noqa: E402

# 40 days of synthetic 1 minute BTC candles moving about 1.5 bp per minute
rng = np.random.default_rng(0)
n = 40 * 1440
close = 100_000 * np.exp(np.cumsum(rng.normal(0, 1.5e-4, n)))
idx = pd.date_range("2026-07-01 04:00", periods=n, freq="1min", tz="UTC")
bars = pd.DataFrame(
    {"open": np.r_[close[0], close[:-1]], "high": close, "low": close, "close": close},
    index=idx,
)
feat = FV.vol_features(bars)

spec = parse_spec(
    "Bitcoin Up or Down - August 9, 10:45AM-11:00AM ET", pd.Timestamp("2026-08-09 15:00", tz="UTC")
)
strike = 100_000.0  # the window's opening TWAP
spot = strike * (1 + 1e-4)  # spot sits 1 bp above it
seconds_left = np.array([120, 90, 60, 45, 30, 15, 5, 1])
t = pd.DatetimeIndex(spec.end - pd.to_timedelta(seconds_left, unit="s"))

# add live rows at the decision instants with spot set to spot like the live recorder does
g = feat.reindex(feat.index.union(t), method="ffill")
g.attrs = feat.attrs
g.loc[t, "S"] = spot

n_known = np.clip(spec.twap_s - seconds_left, 0, None).astype(float)  # prints already in the average
p_twap = FV.fair_value(spec, g, t, base=strike, twap_known=(n_known * spot, n_known))
point = spec.__class__(**{**spec.__dict__, "twap_s": 0})
p_point = FV.fair_value(point, g, t, base=strike)

print(f"{spec.symbol}  {spec.start:%H:%M}-{spec.end:%H:%M} UTC  TWAP {spec.twap_s}s  spot = strike + 1 bp\n")
print(f"{'s left':>6}  {'known':>5}  {'P(up) TWAP':>10}  {'P(up) point':>11}  {'gap':>6}")
for s, k, a, b in zip(seconds_left, n_known, p_twap, p_point):
    print(f"{s:>6}  {k / spec.twap_s:>5.0%}  {a:>10.3f}  {b:>11.3f}  {100 * (a - b):>5.1f}c")
