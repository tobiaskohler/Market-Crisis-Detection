# Random-Forest based Market Crisis Detection 

## Introduction
This repo aims to detect potential market crisis. It utilizes the power of machine learning (random forests) and is fed with a bunch of economic and financial indicators. 

The output of the algorithm is similiar to a traffic light and can be used in portfolio management to allocate assets based on predicted market state.

> **green**: allocation to risky assets is favorable

> **yellow**: allocation to risky assets should be done with caution / portfolio insurance / hedging measures should be impelemented / proportion of risky assets should be reduced

>**red**: only allocate to cash / risk-free rate

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

## Implementation

1. Data cleansing

2. Training

3. Verification

4. Testing

## Examples



