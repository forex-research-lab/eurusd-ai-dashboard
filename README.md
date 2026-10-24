# EUR/USD AI Research — iPhone Safari edition

A mobile-friendly **historical research** web application built with Streamlit. It does not connect to a broker or execute orders. It is not a verified profitable model and does not provide live trading advice.

## Try on a computer (optional)

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Put it online so it works on iPhone (no Python installation on the phone)

You need a free GitHub account and an account with a compatible Streamlit hosting service. Hosting availability, quotas, and free tiers can change.

1. Unzip this project on a computer **or** upload its individual files through the GitHub website in Safari. Preserve `data/` and `.streamlit/` folders.
2. Create a **private** GitHub repository if you do not want the code or uploaded files public. Note: some hosting plans require public repositories; check before uploading any private data. This project includes only fabricated demo data.
3. Upload `dashboard.py`, `research.py`, `requirements.txt`, `README.md`, `.streamlit/config.toml`, and `data/synthetic_TEST_ONLY.csv` to the repository.
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Select **New app**, select the repository, branch `main`, and entrypoint `dashboard.py`, then deploy. The hosting provider will install packages and provide a web address. If your plan does not support the repository's visibility, use a supported repo or plan.
6. Open the resulting web address in Safari on your iPhone. Use Share > Add to Home Screen if you want an app-like shortcut.
7. In the sidebar, choose **Synthetic demonstration** to explore the interface or **Upload EUR/USD CSV** to analyse genuine historical 5-minute data.

No live market feed or historical EUR/USD market dataset is bundled. The included CSV is **synthetic**, explicitly labelled, and only intended to test the UI.

## Required data

A CSV with `timestamp,open,high,low,close` and optionally `spread_pips`. One row per **completed five-minute candle**; timestamp is the candle closing time. UTC is recommended. Data should cover several months to years for meaningful analysis. Upload limits depend on the hosting plan.

## Outputs

- Historical candlestick chart and candidate entry markers
- Logistic regression probability of a +1 ATR target being hit before a -1 ATR stop, within 12 bars
- Chronological 60/20/20 train/validation/test split with 12-bar embargo
- Brier score, ROC AUC, calibration chart
- Fixed-cost backtest with trade table, net expectancy and drawdown
- Downloadable metrics JSON, candidates CSV, and selected trades CSV

## Caveats

- Only OHLC data: intrabar target/stop ordering is uncertain. Stop is prioritised on collisions.
- Fixed round-trip trading costs are a sensitivity assumption, not broker fills.
- Demo prices are synthetic; do not infer predictive skill from them.
- Probability slider changes based on test results are exploratory and invalidate a claim that the selected threshold was tested untouched.
- Some signals overlap; the trade report prevents overlapping open positions but statistical uncertainty and market regimes still matter.
- No economic news filter, broker-specific bid/ask execution, or live order execution.
- The app is not secured as a personal financial-data vault; do not upload sensitive account information.
