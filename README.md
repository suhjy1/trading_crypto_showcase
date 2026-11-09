# Pricing short-dated crypto binaries on Polymarket

> **Note:** This is a condensed version of a larger private research codebase, trimmed to the pieces that show the pricing model and evaluation. Some write-ups refer to files that are not included here. Feel free to reach out if you have questions or want more details.

## Introduction

Polymarket lists 15-minute and hourly "Bitcoin Up or Down" contracts on BTC, ETH, SOL and XRP. Each contract pays $1 if
the price at the end of the window is at or above the price at the start. The underlying assets trade on exchanges that
are much faster than the prediction market, so the natural question is whether a market maker or taker pricing off
those faster feeds could earn money after fees and adverse selection. To answer this, I built a pricing model, a
market-data recorder and an order-book simulator.

The short answer is no. None of the strategies beat the market after realistic execution. The more useful results are
how I established that, and what it showed about where the edge goes. This repository contains the pricing model and
the evaluation code. The rest of the infrastructure is described below but not included. This was research only, and no strategy here was traded with real money.

The study period was 2026-08-01 to 2026-11-05. Data from 2026-11-06 onward was held out.

## The contract

The two tenors settle differently, and this ended up mattering a lot for pricing. The hourly contract settles on a
Binance 1-minute candle, open vs close, which makes it a standard European digital. The 15-minute contract settles on
Chainlink's 60-second TWAP at both the start and the end of the window, which makes it an Asian digital with an
uncertain strike.

| Tenor | Settles on | Pricing problem |
|---|---|---|
| Hourly | Binance 1-minute candle, open vs close | European cash-or-nothing digital |
| 15-minute | Chainlink 60-second TWAP, start and end | Arithmetic Asian digital with an uncertain strike |

Since the windows are only minutes long, drift and discounting are below 1e-6. Pricing therefore comes down to two
things: forecasting the variance to settlement, and modeling the distribution of standardized moves.

## The model

    P(up) = F( (ln(E/K) − V/2) / √V )

V is the variance to settlement. I forecast it with a seasonal EWMA, which is a mean-reverting variance level on
deseasonalized 1-minute returns. That level is multiplied by an intraday profile (weekday/weekend × ET hour) and
integrated exactly over fractional minutes ([`forecast_var`, `season_sum`](updown/fair_value.py)).

For TWAP settlement, E and V have to account for the prints already inside the averaging window, since those are known.
The remaining variance then scales with the cube of the window share still to come
([`twap_moments`](updown/fair_value.py)).

F sets the tails. I use the empirical CDF of past standardized moves for each time-to-expiry bucket, stored at 2,001
quantiles and refit daily on past data only ([`fit_cdfs`, `fair_value_cal`](updown/fair_value.py)).

Contract terms (payoff type, window, and TWAP vs point settlement) are parsed from each market's title and rules text
([`parse_spec`, `settle`](updown/spec.py)). Greeks are the closed-form digital delta, gamma and vega, scaled by the
share of the TWAP that is not yet fixed ([`digital_greeks`](updown/greeks.py)).

The TWAP term matters most near expiry. Take a contract where spot sits 1 bp above the strike. With 30 to 45 seconds
left, a quarter to half of the average is already fixed, and a point-price model is about 19c off
([`examples/twap_vs_point.py`](examples/twap_vs_point.py)).

```
s left  known  P(up) TWAP  P(up) point     gap
    60     0%       0.861        0.734   12.7c
    45    25%       0.953        0.765   18.7c
    30    50%       0.999        0.812   18.7c
    15    75%       1.000        0.895   10.5c
```

Exhibit A: P(up) for a 15-minute contract with spot held 1 bp above the strike, priced once as a TWAP digital and once
as a point digital.

## Results

### Variance forecasting

I compared seven variance models on 305k forecasts across 4 assets, 90 days, and horizons of 5, 15 and 60 minutes.
Each model was scored by QLIKE in paired, day-clustered tests with weekly walk-forward refits. GARCH(1,1) was the only
model to beat the seasonal EWMA, and only at the 15-minute horizon (QLIKE −46 ×1e3, t −2.1). HAR was significantly
worse at short horizons (+129, t 9.2). HAR had been running in my paper market maker, and I rolled it back on this evidence.

### Distribution shape

To isolate the effect of the tails, I held the state z fixed and varied only the CDF, over 58 walk-forward days and
9.2M market-seconds. The empirical CDF was the only candidate that beat the incumbent in every contract class, and
Gaussian tails were clearly the worst on the 15-minute contracts. However, the traded price beat every distribution by
6 to 14 log-loss points (×1e3), while better tails were only worth 0.5 to 1.6. The market's advantage is information,
not tail shape.

### Settlement-source reconstruction

Before testing any strategy, I checked that I could reproduce the settlement price myself. The mean of Chainlink
prints over the final 60 seconds reproduces 99–100% of official outcomes, compared to 85–94% for the point print. I
then built a nowcast of the settlement price from the faster exchange feeds. It has a mean error of 0.5–0.6 bp, and in
the final 10 seconds it calls 98–100% of outcomes.

### Market-anchored residual

Since the market price beat every standalone model, I next asked whether the model could add anything on top of it.
I fitted `logit p = logit p_mkt + g(X)`, and the model ends up with about 10% weight against the market. The one
stable exception is the final minute of 15-minute contracts, where the TWAP is partly known (−1.06 log loss ×1e3,
t −4.0, after I fixed a variance bug that counted partial minutes as whole minutes; −2.9 before the fix).

### Options-implied pricing: a Deribit volatility surface

Polymarket and Kalshi also list daily "above $K" ladders, and Deribit lists BTC and ETH options on the same underlying. I wanted
to see if the options market could price these ladders better than the prediction markets do
([notebook](notebooks/vol_surface.ipynb), [`surface.py`](updown/surface.py)).

I fitted an arbitrage-free SSVI surface to Deribit BTC and ETH options, with one (ρ, φ) per expiry. θ is
non-decreasing by construction, and φ is capped at the butterfly-free bound. The fit lands inside the bid/ask on 70–76%
of options, with an RMSE under one vol point.

Deribit's daily options expire at 08:00 UTC, so the ladders settle between two expiries. Rather than interpolating by
clock time, I interpolated in variance time, reusing the intraday profile from the realized-vol model. On a weekend
target, this moves the variance share from 37% (flat clock) to 23%.

Digitals come from −∂C/∂K, which adds a skew term worth 4–7c at the money. Against a live Polymarket ladder, the venue
sat about 3c from the surface, which was closer than the naive N(d2) at about 5c. The largest gaps were at the money,
where the USDT/USD basis dominates.

I parked this line of work. Deribit's shortest expiry is daily, so the 15-minute and hourly contracts would need the
front smile extrapolated down to minutes. At that horizon, spot and the settlement basis matter more than the smile.

### Where a market maker's edge goes

To see where the edge went, I decomposed P&L fill by fill ([`decompose.py`](updown/decompose.py)). A maker replay
carried +6c of model edge at the decision and gave back −7c by settlement. In other words, the model's price was wrong
by more than the half-spread on exactly the fills it got. I also ran two pre-registered maker strategies through an
event-driven simulator calibrated against on-chain fills, and both failed (−1.17c per share, t −3.8; −0.87c, t −1.2).

### Negative results

I want to be clear about what did not work.

1. The market price beats every standalone model.
2. Taking the model's gaps against real prints loses 3.4c per share on 15-minute contracts, before fees.
3. Edges that appeared with proxy inputs (Binance standing in for Chainlink) vanished once I used the observed inputs.
4. A 3-second input delay costs +35 Brier ×1e4 on 15-minute contracts (t 10.1). On 5-minute contracts this turns into
   a latency race rather than a modeling problem, so I set them aside.

## Evaluation standards

Every result above was held to the same standards, and the inference code is in [`stats.py`](updown/stats.py). All
comparisons are paired on identical rows, with day- or market-clustered standard errors. Fills within a day are
correlated, so naive t-stats can overstate significance several-fold.

Strategy results get a wild cluster bootstrap verdict. A result only counts as POSITIVE if its lower bound stays above
zero after dropping any single day, and after dropping the two best days. Hypotheses and pass rules were pre-registered,
the held-out period was never read, and I applied Holm corrections across cells.

One thing I was careful about throughout was look-ahead. Every feature is gated on when its input was received. The
tests perturb all data after t and check that the price at t does not change
([`test_no_look_ahead`](tests/test_fair_value.py)).

## The wider system (not in this repo)

The full project was about 16k lines of package code plus 16k lines of research scripts, with 302 offline tests. Only
the pricing and evaluation pieces are included here.

The market-data recorder runs 24/7, recording Polymarket order books, the settlement price feeds, perpetual and spot
quotes from 6 exchanges, and Kalshi. Every row carries both the venue's timestamp and the local receipt timestamp. The
data goes into an append-only Parquet lake with a SQLite catalog, along with 27M on-chain fills backfilled from Polygon.
The order-book simulator replays recorded events in receipt order against simulated orders, with price-time queue
position and latency timers. Its fill model is calibrated against the on-chain fills.

I also studied the other market makers, using fills from about 20k on-chain wallet clusters. Skill persists, and it is
concentrated on the losing side. Finally, I audited the equivalent contracts on Polymarket and Kalshi, and studied
trading both sides across the two venues.

## Code in this repo

```
updown/
  spec.py          contract terms from title and rules text; settlement from candles
  fair_value.py    variance forecasts, TWAP moments, calibrated tails, fair value
  greeks.py        digital greeks and stale-quote (pickoff) cost
  decompose.py     per-fill P&L decomposition (identity holds row by row)
  stats.py         clustered t, wild cluster bootstrap, pre-registered verdict rule
  surface.py       Deribit implied vols, SSVI fit, arbitrage checks, variance clock, digitals
notebooks/
  vol_surface.ipynb  the options-surface study, outputs from a 2026-11-07 snapshot
examples/
  twap_vs_point.py the TWAP effect on synthetic data
data/
  fetch.py         downloads the Deribit snapshot and BTC variance profile the notebook reads (public APIs)
tests/             50 tests: pricing identities, exact fractional variance integrals,
                   cubic TWAP scaling, no look-ahead, P&L identity, inference verdicts,
                   SSVI recovery, digital = −dC/dK, no-arbitrage constraints
```

## Running

```
pip install -e ".[test,notebook]"
python -m pytest
python examples/twap_vs_point.py
python data/fetch.py
jupyter lab notebooks/vol_surface.ipynb
```

The tests and the example run on synthetic data. The notebook needs a Deribit snapshot, which `python data/fetch.py`
downloads into `data/`. The outputs saved in the notebook come from a snapshot taken on 2026-11-07. Raw market data is
not included in this repository.

## License

The code is MIT licensed (see [LICENSE](LICENSE)). The license covers the code only. Third-party market data (Deribit,
Binance, Chainlink, Polymarket, Kalshi) is not included and remains under its providers' terms.
