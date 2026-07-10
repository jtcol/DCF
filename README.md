# 📈 Equity Toolkit — DCF Valuation + LEAPS Screener

A professional three-tab Streamlit app:

- **📈 DCF Valuation** — estimate a company's per-share fair value with an unlevered (FCFF)
  discounted cash flow model, using live Yahoo Finance data and editable smart defaults.
- **🎯 LEAPS Screener** — scan the US market for long-dated call (LEAPS) candidates: deeply
  oversold names still in an uptrend, with cheap option premium.
- **🌊 Volatility Risk Premium** — rank Nasdaq-100 / S&P 500 / watchlist tickers by how *rich*
  their option premium is (30-day implied vol vs 30-day realized vol) — an option-seller's edge.

> ⚠️ Educational / research decision-support tool only. **Not investment advice.**

---

## DCF Valuation — features

- **Unlevered FCFF DCF** — projects Free Cash Flow to the Firm, discounts at WACC, subtracts
  net debt to reach equity value and a per-share fair value.
- **Auto-calculated WACC (CAPM)** with full manual override of every component.
- **Revenue-driven, 2-stage projection** with optional growth fade toward the terminal rate.
- **Dual terminal value** — Gordon Growth perpetuity **and** Exit Multiple (EV/EBITDA), shown
  side-by-side and averaged to reduce single-method bias.
- **Smart defaults** auto-derived from each company's own history (revenue CAGR, FCF margin,
  effective tax rate, implied EV/EBITDA), all editable.
- **Reverse DCF** — solves for the growth rate the current market price implies, so you can
  sanity-check whether your assumptions are more or less optimistic than the market.
- **Historical financials & charts** plus a **Data Quality** panel that flags missing fields,
  short history, negative/volatile FCF, and the financial-sector FCFF caveat.
- **Ticker selection** via a searchable S&P 500 dropdown *and* a free-text box for any ticker.

---

## LEAPS Screener — features

Scans the US market with a **cheapest-first funnel** so expensive options calls only run on the
few survivors:

1. **Universe** — US stocks above a market-cap / volume / price floor (default cap > $2B,
   volume > 1M, price > $5), sourced from the NASDAQ screener API in one call.
2. **Weekly RSI < 25** — deeply oversold on a slow timeframe (a pullback, not a chase).
3. **Ripster EMA cloud bullish stack** (daily) — fast cloud (EMA 5/12) above med (34/50) above
   slow (72/89), no overlap → the longer-term trend is still up.
4. **ATM IV < 30%** — cheap option premium (fetched only for survivors).

Each candidate gets an **IV-percentile proxy**, a **suggested LEAPS contract** (~12-month+ expiry,
~0.80-delta deep-ITM call chosen via Black-Scholes delta), liquidity (spread & open interest),
earnings proximity, and a composite **LEAPS Score (0–100)**. Includes a "how to pick a great
LEAPS" guidance panel and CSV export.

---

## Volatility Risk Premium — features

Ranks tickers by option-premium richness across **Nasdaq-100**, **S&P 500**, or a manual
**watchlist**. Per ticker it computes:

1. **IV30** — at-the-money 30-day implied vol, variance-interpolated to exactly 30 DTE.
2. **RV30** — annualized realized vol over the last ~21 trading days.
3. **VRP = IV30 − RV30** and the **IV/RV ratio** (> 1 ⇒ "Good (rich)" — options priced richer
   than the stock is actually moving; favorable for premium sellers).
4. **IV-rank proxy** → *Rich* > 50 / *Cheap* < 30 (current IV vs the stock's 1-yr realized-vol
   range).

Results are **ranked best→worst by IV/RV**, with a guidance panel and CSV export.

---

## Project structure

```
DCF/
  app.py                 # thin entry: page config + three top-level tabs
  dcf/
    dcf_tab.py           # render_dcf_tab() — the full DCF dashboard
    leaps_tab.py         # render_leaps_tab() — the LEAPS screener UI
    vrp_tab.py           # render_vrp_tab() — the Volatility Risk Premium UI
    data.py              # yfinance fetch + cleaning + smart-default assumptions
    model.py             # WACC, FCFF projection, terminal value, fair value
    reverse_dcf.py       # implied-growth solver (bisection)
    sp500.py             # S&P 500 constituent list (live + fallback)
    nasdaq100.py         # Nasdaq-100 constituent list (live + fallback)
    quality.py           # data-quality flags
    formatting.py        # currency / percent / big-number formatting
  leaps/
    universe.py          # US universe via NASDAQ screener API (+ cap/vol/price filters)
    indicators.py        # weekly RSI + Ripster EMA clouds / bullish-stack detection
    options_iv.py        # ATM IV, IV-percentile proxy, Black-Scholes LEAPS suggestion
    prices.py            # shared batched price-history download
    screener.py          # staged LEAPS scan funnel + LEAPS score
    vrp.py               # VRP engine: IV30, RV30, IV/RV ratio, ranking
  data/sp500_fallback.csv
  data/nasdaq100_fallback.csv
  requirements.txt
  .env                   # config (git-ignored)
  .gitignore
```

---

## Setup & run (local)

```bash
cd DCF
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. Pick a ticker, review the pre-filled assumptions, and click
**Generate Valuation**.

---

## Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing at `DCF/app.py`.
3. (Optional) Add the variables from `.env` under the app's **Secrets** settings.

Streamlit Cloud auto-redeploys on every push to the connected branch.

---

## Methodology notes

**FCFF per year** = `EBIT × (1 − tax) + D&A − Capex − ΔWorking Capital`. In projection, FCFF is
modelled as `Revenue × FCF margin`, where revenue grows at the stage-1 rate (optionally fading to
the terminal rate).

**WACC (CAPM):**
- Cost of equity = `risk-free + β × equity risk premium`
- After-tax cost of debt = `pre-tax cost of debt × (1 − tax)`
- WACC = `equity weight × cost of equity + debt weight × after-tax cost of debt`
- Risk-free rate defaults to the 10-Year Treasury yield (`^TNX`).

**Terminal value:**
- Gordon Growth: `FCFF_{N+1} / (WACC − g)`
- Exit Multiple: `Terminal EBITDA × EV/EBITDA multiple`
- The app averages whichever methods you enable.

**Fair value** = `(PV of explicit FCFF + PV of terminal value − net debt) / shares outstanding`.

### LEAPS screener methodology

- **Weekly RSI** = 14-period RSI (Wilder smoothing) on weekly bars resampled from daily closes.
- **Ripster clouds** = EMAs 5/12 (fast), 34/50 (med), 72/89 (slow). Bullish stack requires the
  fast cloud entirely above the med cloud, and the med cloud entirely above the slow cloud.
- **ATM IV** = average of the at-the-money call & put implied vol from the nearest expiry ≥ 30 DTE.
- **Suggested LEAPS** = first expiry ≥ ~300 DTE, and the call whose **Black-Scholes delta** is
  closest to the target (default 0.80). Greeks are computed locally (yfinance chains omit them).
- **LEAPS Score (0–100)** = weighted blend of oversold-ness, EMA-stack separation, low ATM IV,
  low IV-percentile proxy, option liquidity (spread & open interest), and distance from earnings.

**Performance:** a full US-market scan is ~6–15 min on a cold run (bulk price download for the
filtered ~1,500–2,500 names) and near-instant on cached re-runs (15-min TTL). Use the *Watchlist*
or *S&P 500* scope, or a universe limit, to test quickly.

### Volatility Risk Premium methodology

- **RV30** = `std(daily log returns, last 21 trading days) × √252` (annualized).
- **IV30** = ATM implied vol interpolated to 30 DTE — linear in *variance* between the two expiries
  bracketing 30 days (ATM = average of the nearest call & put IV per expiry).
- **VRP** = `IV30 − RV30`; **IV/RV** = `IV30 / RV30`. **Good (rich)** premium when IV/RV > 1.
- **IV regime** from the IV-rank proxy: **Rich** > 50, **Cheap** < 30, else Neutral.
- Tickers are sorted by **IV/RV descending**. Every ticker needs an options fetch, so runs are
  threaded (~8 workers): Nasdaq-100 ≈ 1–3 min, S&P 500 ≈ 5–15 min, watchlist seconds.
- Comparing *forward* IV to *trailing* RV is the standard practical VRP screen (true VRP compares
  IV to subsequently-realized vol, which isn't knowable in advance).

---

## Accuracy & limitations

- `yfinance` is an **unofficial** Yahoo Finance scraper: it can be rate-limited, delayed, or
  missing fields. The app caches results, falls back gracefully, and flags gaps in **Data Quality**.
- DCF output is highly sensitive to assumptions — especially WACC and terminal growth. Watch the
  **"terminal value % of EV"** metric; when it dominates, treat the number with caution.
- Unlevered FCFF DCF is a **poor fit for banks, insurers, and REITs** (financing-driven cash
  flows). These are flagged, not blocked.
- Always cross-check the model against the **reverse DCF** and the company's actual history.
- **LEAPS "IV Rank/Percentile" is a proxy.** yfinance provides no historical implied volatility,
  so the screener ranks current IV against the stock's trailing **1-year realized-volatility**
  range — a stand-in for true IV Rank, clearly labeled as such.
- **Weekly RSI < 25 combined with a bullish EMA stack is rare** — a full scan may return few or
  zero names. Loosen the RSI threshold (e.g. 30–35) to widen results.
- Options chains from yfinance can be thin or stale; always verify a contract in your broker
  before trading.
