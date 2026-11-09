"""
Splits each fill's P&L into additive parts that sum to the realized P&L, row by row.

decision_edge: model value at decision vs my quoted price.
decision_to_fill: model drift between the decision and the fill.
post_fill_markout: prevailing mid at a horizon vs model value at the fill.
settlement_residual: outcome vs that mid.
fee_term: fees and rebates.

I use this to see where a strategy's edge goes, whether the model was wrong, the fill was adversely selected, or the market moved after the fill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS = (
    "decision_edge",
    "decision_to_fill",
    "post_fill_markout",
    "settlement_residual",
    "fee_term",
)
HORIZON_PREF = ("60s", "30s", "5s", "1s", "250ms")


def components(f: pd.DataFrame, horizon: str = "60s") -> pd.DataFrame:
    s = np.where(
        f["side"].astype(str).str.lower().isin(["buy", "yes", "bid"]), 1.0, -1.0
    )
    size = f["size"].to_numpy(float)
    px = f["price"].to_numpy(float)
    y = f["outcome"].to_numpy(float)
    fee = np.nan_to_num(f["fee"].to_numpy(float) if "fee" in f else np.zeros(len(f)))
    theo_dec = f["theo_dec"].to_numpy(float)
    theo_fill = f["theo_fill"].to_numpy(float)
    theo_dec = np.where(np.isnan(theo_dec), theo_fill, theo_dec)
    theo_fill = np.where(np.isnan(theo_fill), theo_dec, theo_fill)
    mark = (
        f[f"mark_mid_{horizon}"].to_numpy(float)
        if f"mark_mid_{horizon}" in f
        else np.full(len(f), np.nan)
    )
    src = (
        f["mark_source"].astype(object).to_numpy()
        if "mark_source" in f
        else np.full(len(f), None)
    )
    mid_h = np.where(np.isnan(mark), np.nan, px + s * mark)
    merged = np.isnan(mid_h) | (src == "outcome")
    # no usable mark so post fill and settlement merge into s * (y - theo_fill)
    mid_h = np.where(merged, theo_fill, mid_h)
    out = pd.DataFrame(
        {
            "decision_edge": s * (theo_dec - px) * size,
            "decision_to_fill": s * (theo_fill - theo_dec) * size,
            "post_fill_markout": s * (mid_h - theo_fill) * size,
            "settlement_residual": s * (y - mid_h) * size,
            "fee_term": -fee,
            "merged": merged,
        },
        index=f.index,
    )
    out["total"] = out[list(COMPONENTS)].sum(axis=1)
    return out


def check_identity(
    f: pd.DataFrame, horizon: str = "60s", tol: float = 1e-6
) -> tuple[bool, float]:
    c = components(f, horizon)
    ok = f["pnl"].notna() & f["outcome"].notna()
    if not ok.any():
        return True, 0.0
    d = (c.loc[ok, "total"] - f.loc[ok, "pnl"].to_numpy(float)).abs().max()
    return bool(d <= tol), float(d)


def aggregate(
    f: pd.DataFrame, by: list[str] | None = None, horizon: str = "60s"
) -> pd.DataFrame:
    c = components(f, horizon)
    d = pd.concat(
        [f[(by or []) + ["size"]].reset_index(drop=True), c.reset_index(drop=True)],
        axis=1,
    )
    d = d[f["outcome"].notna().to_numpy()]
    g = d.groupby(by, observed=True) if by else d.assign(_all="all").groupby("_all")
    rows = []
    for key, x in g:
        n = x["size"].sum()
        row = {
            "fills": len(x),
            "contracts": round(n),
            "merged_share": float(x["merged"].mean()),
        }
        for k in (*COMPONENTS, "total"):
            row[f"{k}_c"] = 100 * x[k].sum() / n if n else np.nan
        row["total_usd"] = x["total"].sum()
        if by:
            row.update(dict(zip(by, key if isinstance(key, tuple) else (key,))))
        rows.append(row)
    cols = (by or []) + [c for c in rows[0] if c not in (by or [])] if rows else []
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame()
