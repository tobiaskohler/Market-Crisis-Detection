# Random Forest vs. HMM vs. Naive

## Introduction
This repo aims to detect potential market regimes. In particular, it compares the capability of a Random Forest model vs. a Gausian Hidden Markov Model. Both models are fed with a bunch of economic and financial indicators (see subsequently).


## Indicators / Features

### Daily
- OFR Financial Stress Index (https://www.financialresearch.gov/financial-stress-index/)
- T10Y2Y Spread (https://fred.stlouisfed.org/series/T10Y2Y)
- T103M Spread (https://fred.stlouisfed.org/series/T10Y3M)
- Wilshire 5000 Index (https://fred.stlouisfed.org/series/WILL5000PRFC)

### Weekly
- AAII Sentiment Survey (https://www.aaii.com/sentimentsurvey/sent_results)

### Monthly
- IFO Geschäftsklimaindex / IFO business climate index (German) (https://www.ifo.de/ifo-zeitreihen)
- US Unemployment rate (https://fred.stlouisfed.org/series/UNRATE)
- US Consumer sentiment (https://fred.stlouisfed.org/series/UMCSENT)

### Quarterly
- US GDP (https://fred.stlouisfed.org/series/GDP)
- EU GDP (https://fred.stlouisfed.org/series/CPMNACSCAB1GQEU272020)
- "Buffet Indicator": ratio of United States stock market (represented by Wilshire 5000) to GDP 

> Indicators not available on a daily basis were updated linearly until a new data point was available.


### Engineered / Synthetic Features
- Date-Related Features (Day of week/Month/Quarter/)
- Lags of daily indicators
- Rolling Weighted average, Max, Min and StdDev of each daily Feature

## Summary in a nutshell
- first and foremost: **no tx costs**
- forecast is for two days ahead (since data for indicators might be published with delay)
- meaning of market lights:
    - green: 100% risky assets (stocks)
    - yellow: 60% stocks/40% bonds
    - red: risk free rate (assumed 0%)
- Samples: 6591, Features: 173
- implemented plain vanilla Random Forest (70% training data, 100 trees, **no cross-validation, no grid-search, etc.**)
- *market_light* is derived from drawdown on a fictive Buy and Hold strategy:
    - 0 > drawdown -> market_light == 1 (green)
    - -0.02 <= drawdown <= 0 -> market_light == 0 (yellow)
    - drawdown <= 0.05 -> market_light == -1 (red, crisis)
- implemented Gausian Hidden Markov Model (GHMM, 100 iterations)
- implemented naive Benchmark strategies: 
    - Buy and Hold
    - SMA 30/200 crossover long-only filter
    - Drawdown-adaptive stragey (involves refraining from investing on the next day if the drawdown on day t falls below -0.05. If the drawdown falls below -0.02, the allocation is adjusted to 60% in stocks and 40% in bonds.)
- GHMM excelled in terms of performance
- Random Forest outperformed naive strategies, but ended up performing lower than simple Buy and Hold strategy
- drawdown of GHHM is significantly lower than with the RF strategy
![](./predictions/BAH_vs_adaptive1680382652.367612.png)


