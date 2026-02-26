# Chart Signal Lab

Chart Signal Lab is a browser-based chart assistant that combines:

1. **Public live/historical market candles**
   - Crypto: Binance klines API
   - Stocks/indices/forex/metals: Stooq daily OHLC API
2. **On-device model training**
   - A lightweight logistic classifier is trained in the browser from recent OHLC-derived features (RSI, SMA distance, EMA spread, ATR, momentum, volume z-score).
3. **Optional uploaded chart image**
   - Image-based trend/volatility features are used as a secondary signal and blended with model probability.

## Outputs

For each analysis run, the app returns:

- BUY/SELL bias
- Confidence
- Model directional probability
- Validation accuracy (holdout split)
- Entry, target, stop loss
- Risk/reward ratio
- Trader narrative/context

## Run locally

```bash
python3 -m http.server 4173
```

Then open: `http://localhost:4173`

## Notes

- This increases signal quality over pure image heuristics by grounding decisions in real market OHLC data.
- It is still probabilistic and for educational use only, not financial advice.
