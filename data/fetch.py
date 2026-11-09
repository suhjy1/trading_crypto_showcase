"""Refresh the data that the vol-surface notebook runs on (public endpoints, no keys needed).

This writes two files next to this one. deribit_options.parquet is a single Deribit option-chain snapshot for BTC and ETH. btc_variance_profile.json is the intraday variance profile (weekday/weekend x ET hour) from the last 40 days of Binance BTCUSDT 1-minute candles, computed the same way as in `fair_value`.

    python data/fetch.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from updown import fair_value as FV  # noqa: E402
from updown import surface as SF  # noqa: E402

KLINES = "https://data-api.binance.vision/api/v3/klines"  # public market-data mirror


def binance_1m(symbol: str, days: int) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="UTC").floor("min")
    t = int((end - pd.Timedelta(days=days)).timestamp() * 1000)
    rows = []
    while t < end.timestamp() * 1000:
        r = requests.get(
            KLINES, params={"symbol": symbol, "interval": "1m", "startTime": t, "limit": 1000}, timeout=30
        ).json()
        if not r:
            break
        rows += r
        t = r[-1][0] + 60_000
    d = pd.DataFrame(rows).iloc[:, :5]
    d.columns = ["ts", "open", "high", "low", "close"]
    d.index = pd.to_datetime(d.pop("ts"), unit="ms", utc=True)
    return d.astype(float).iloc[:-1]  # drop the candle still open


def main() -> None:
    chain = pd.concat([SF.fetch_snapshot(c) for c in ("BTC", "ETH")], ignore_index=True)
    chain.to_parquet(HERE / "deribit_options.parquet", index=False)
    print(f"deribit: {len(chain)} options, as of {chain['ts'].iloc[0]:%Y-%m-%d %H:%M} UTC")

    bars = binance_1m("BTCUSDT", 40)
    prof = FV.vol_features(bars).attrs["profiles"]
    day = max(prof)
    out = {"symbol": "BTCUSDT", "profile_date": str(day.date()), "profile": np.round(prof[day], 6).tolist()}
    (HERE / "btc_variance_profile.json").write_text(json.dumps(out, indent=1))
    print(f"profile: {len(bars)} candles, profile date {day.date()}")


if __name__ == "__main__":
    main()
