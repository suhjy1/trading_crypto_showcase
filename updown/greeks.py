"""
Closed-form greeks of a digital in log-moneyness, and what a stale quote costs.

a is the share of the TWAP still to come. As the average fixes, a shrinks, and delta and gamma shrink with it.

pickoff takes delta, gamma and a reaction latency, and returns the expected first- and second-order fair value move that a resting quote is exposed to.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def d_of(lm, V) -> np.ndarray:
    V = np.asarray(V, float)
    return (np.asarray(lm, float) - V / 2) / np.sqrt(V)


def digital_greeks(lm, V, a=1.0) -> dict[str, np.ndarray]:
    V = np.asarray(V, float)
    a = np.asarray(a, float)
    d = d_of(lm, V)
    sq = np.sqrt(V)
    ph = norm.pdf(d)
    return {
        "p": norm.cdf(d),
        "delta": a * ph / sq,
        "gamma": -(a**2) * d * ph / V,
        "vega": -ph * (d + sq),
    }


def pickoff(delta, gamma, s1, lat_s) -> tuple[np.ndarray, np.ndarray]:
    var = np.asarray(s1, float) ** 2 * lat_s
    return np.abs(delta) * np.sqrt(var), 0.5 * np.abs(gamma) * var
