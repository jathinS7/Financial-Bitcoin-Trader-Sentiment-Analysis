# Bitcoin Trader Sentiment Analysis

Does market sentiment explain trader performance? This project joins daily Crypto Fear & Greed Index values to individual trade records, then tests whether win rate and PnL differ between sentiment regimes. It also trains a Random Forest to predict winning trades and clusters accounts by trading behaviour.

## Method

1. Parse trade timestamps (`%d-%m-%Y %H:%M`) and join each trade to that day's sentiment value and label.
2. Keep closed trades only (`Closed PnL != 0`, real trade directions only).
3. Compare win rate and PnL percentage by sentiment class. PnL percentage is capped at the 1st and 99th percentiles to limit outliers.
4. Run a two-proportion z-test comparing win rates in "Extreme Greed" days against all other days.
5. Train a Random Forest (250 trees, max depth 8, balanced class weights) on trade-level features, with a stratified 80/20 split.
6. Cluster accounts with KMeans (k = 3) on size, win rate, fee and taker ratio.

## Data

The script reads two CSVs from the same folder. They are **not included** in this repo:

- `fear_greed_index.csv`: columns `date`, `value`, `classification`
- `historical_data.csv`: trade records with `Timestamp IST`, `Coin`, `Direction`, `Size USD`, `Closed PnL`, `Fee`, `Crossed`, `Account`

## Run it

```bash
pip install -r requirements.txt
python bitcoin_analysis.py
```

The script prints its results to the console. It does not save figures.

## Project structure

```
bitcoin_analysis.py   # full analysis pipeline
requirements.txt      # dependencies
```

## Limitations

- Results are correlational. Sentiment does not prove a causal effect on trader outcomes.
- The win/loss label uses realised PnL on closed trades only, so open positions are excluded.
- Account clusters depend on the chosen k = 3, which has not been validated (for example with silhouette scores).

## Author

Chillagattu Jathin · [LinkedIn](https://www.linkedin.com/in/jathins7) · [Portfolio](https://jathindsprojects.netlify.app/)
