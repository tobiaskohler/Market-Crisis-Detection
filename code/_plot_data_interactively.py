import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import 00-SP500.csv as pandas df
df = pd.read_csv('../prepared_data/daily/00-SP500.csv', delimiter=';')
df.set_index('Date', inplace=True)

df['returns'] = df['^GSPC'].pct_change()
df['cum_returns'] = (1 + df['returns']).cumprod()

data = df[['^GSPC', 'cum_returns']]
# print(data.tail(10))

# calculate rolling drawdown, period 30 days
# initialize rolling_max with value of first observation
data['rolling_max'] = data['cum_returns'].rolling(window=200, min_periods=1).max()
data['drawdown'] = data['cum_returns']/data['rolling_max'] - 1.0

# #calcuate 200 day moving average
# data['200d'] = data['^GSPC'].rolling(window=200, min_periods=1).mean()
# data['30d'] = data['^GSPC'].rolling(window=30, min_periods=1).mean()

# print(data.tail(10))

drawdown_threshold_green = 0.0
drawdown_threshold_yellow = -0.01
drawdown_threshold_red = -0.05

# create new column with market light
# if drawdown is above drawdown_threshold_green, market light is green
# if drawdown is below drawdown_threshold_red, market light is red
# if drawdown is between drawdown_threshold_green and drawdown_threshold_yellow, market light is yellow

#data['market_light'] =  data['drawdown'].apply(lambda x: 'green' if x > drawdown_threshold_yellow else 'yellow' if x > drawdown_threshold_red else 'red')

##########
# LABELING
##########
data['market_light'] = 1

# GFC (GLOBAN FINANCIAL CRISIS)
data['market_light'].loc['2007-06-01':'2009-03-10'] = -1
#90 days before 2007-06-01 switch to yellow
data['market_light'].loc['2007-03-01':'2007-06-01'] = 0

# COVID19
data['market_light'].loc['2020-02-06':'2020-03-23'] = -1
#90 days before 2020-02-19 switch to yellow
data['market_light'].loc['2019-11-29':'2020-02-05'] = 0

# INTEREST RATE INCREASE
data['market_light'].loc['2021-11-08':'2022-10-14'] = -1
#90 days before 2021-11-08 switch to yellow
data['market_light'].loc['2021-07-08':'2021-11-07'] = 0



# MINOR DRAWDOWNS
data['market_light'].loc['2018-08-30':'2018-12-20'] = -1
#90 days before 2018-08-30 switch to yellow
data['market_light'].loc['2018-01-10':'2018-08-29'] = 0

data['market_light'].loc['2015-02-24':'2016-02-10'] = -1
#90 days before 2015-02-24 switch to yellow
data['market_light'].loc['2014-11-10':'2015-02-23'] = 0

data['market_light'].loc['2010-04-14':'2010-07-07'] = -1
#90 days before 2010-04-14 switch to yellow
data['market_light'].loc['2010-01-10':'2010-04-13'] = 0

data['market_light'].loc['2011-05-02':'2011-10-03'] = -1
data['market_light'].loc['2011-02-17':'2011-05-01'] = 0

data['market_light'].loc['2002-11-30':'2003-03-13'] = -1
data['market_light'].loc['2002-10-17':'2002-11-29'] = 0


# calculate rolling drawdown on a period of 30 days
# initialize rolling_max with value of first observation
data['rolling_max'] = data['cum_returns'].rolling(window=90, min_periods=1).max()
data['drawdown'] = data['cum_returns']/data['rolling_max'] - 1.0

# calculate volatility of drawdown (standard deviation)
#data['drawdown_stddev'] = data['drawdown'].rolling(window=7, min_periods=1).std() # seemed to be ok but maybe to much noise

# calculate max drawdown in a period of 30 days
data['drawdown_max'] = data['drawdown'].rolling(window=30, min_periods=1).min()
# calculate volatility of max drawdown (standard deviation)
data['drawdown_min'] = data['drawdown'].rolling(window=30, min_periods=1).max()
data['drawdown_spread'] = data['drawdown_max'] - data['drawdown_min']

#calculate 7 day moving average and 21 day moving average of drawdown_spread
data['drawdown_spread_1'] = data['drawdown_spread'].rolling(window=30, min_periods=1).mean()
data['drawdown_spread_2'] = data['drawdown_spread'].rolling(window=7, min_periods=1).mean()

#calculate max drawdown spread in a period of 30 days
data['drawdown_spread_max'] = data['drawdown_spread'].rolling(window=30, min_periods=1).max()


# save market_light to csv
data['market_light'].to_csv('../prepared_data/99-LABELS.csv', header=True, sep=";")

plt.set_style='dark_background'
fig, ax = plt.subplots()    
ax.plot(data['^GSPC'], color='black')
ax.fill_between(data.index, data['^GSPC'], where=data['market_light']==-1, color='red', alpha=0.5)
ax.fill_between(data.index, data['^GSPC'], where=data['market_light']==0, color='yellow', alpha=0.5)
ax.fill_between(data.index, data['^GSPC'], where=data['market_light']==1, color='green', alpha=0.5)
#plot drawdown on second y-axis
ax2 = ax.twinx()
# ax2.plot(data['drawdown_max'], color='black', alpha=0.5)
# ax2.plot(data['drawdown_min'], color='blue', alpha=0.5)
# plot drawdown_augmented_market_light only for values == -1
ax2.fill_between(data.index, data['^GSPC'], where=data['drawdown_augmented_market_light']==-1, color='blue', alpha=0.5)

plt.show()
