# 📈 S&P 500 DCF Valuation App

A professional **Discounted Cash Flow (DCF)** tool to estimate the fair value of S&P 500
companies. Financial data is pulled live from Yahoo Finance (via `yfinance`), you confirm a
short list of mandatory drivers in a Streamlit UI, and the app produces a per-share fair value
plus diagnostics that help you judge how much to trust the result.

> ⚠️ Educational / research decision-support tool only. **Not investment advice.**

---

## Features

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

## Project structure

```
DCF/
  app.py                 # Streamlit entry point (sidebar inputs + output tabs)
  dcf/
    data.py              # yfinance fetch + cleaning + smart-default assumptions
    model.py             # WACC, FCFF projection, terminal value, fair value
    reverse_dcf.py       # implied-growth solver (bisection)
    sp500.py             # S&P 500 constituent list (live + fallback)
    quality.py           # data-quality flags
    formatting.py        # currency / percent / big-number formatting
  data/sp500_fallback.csv
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

---

## Accuracy & limitations

- `yfinance` is an **unofficial** Yahoo Finance scraper: it can be rate-limited, delayed, or
  missing fields. The app caches results, falls back gracefully, and flags gaps in **Data Quality**.
- DCF output is highly sensitive to assumptions — especially WACC and terminal growth. Watch the
  **"terminal value % of EV"** metric; when it dominates, treat the number with caution.
- Unlevered FCFF DCF is a **poor fit for banks, insurers, and REITs** (financing-driven cash
  flows). These are flagged, not blocked.
- Always cross-check the model against the **reverse DCF** and the company's actual history.
