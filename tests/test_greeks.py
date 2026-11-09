import numpy as np

from updown.greeks import digital_greeks, pickoff


def _p(lm, V, a=1.0):
    return digital_greeks(lm, V, a)["p"]


def test_greeks_match_finite_differences():
    lm, V, a, h = (
        np.array([-3e-4, 0.0, 2e-4, 8e-4]),
        np.array([4e-8, 1e-7, 2.5e-7, 1e-6]),
        0.6,
        1e-7,
    )
    g = digital_greeks(lm, V, a)
    # moving ln S by h moves ln E by a * h
    up, dn = _p(lm + a * h, V), _p(lm - a * h, V)
    np.testing.assert_allclose(g["delta"], (up - dn) / (2 * h), rtol=1e-4)
    np.testing.assert_allclose(
        g["gamma"], (up - 2 * g["p"] + dn) / h**2, rtol=1e-3, atol=1e3
    )
    x = 1e-5  # scaling sigma by 1 + x scales V by (1 + x)^2
    vg = (_p(lm, V * (1 + x) ** 2) - _p(lm, V * (1 - x) ** 2)) / (2 * x)
    np.testing.assert_allclose(g["vega"], vg, rtol=1e-4, atol=1e-9)


def test_gamma_zero_at_the_money_and_pickoff_scaling():
    V = 1e-7
    g = digital_greeks(V / 2, V)  # d = 0
    assert abs(g["gamma"]) < 1e-6 and g["delta"] > 0
    f1, c1 = pickoff(g["delta"], g["gamma"], 1e-4, 1.0)
    f4, _ = pickoff(g["delta"], g["gamma"], 1e-4, 4.0)
    np.testing.assert_allclose(f4, 2 * f1)
