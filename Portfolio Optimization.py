############ Monte Carlo Simulation for Optimization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

##### getting the data and formatting
aapl = pd.read_csv('AAPL_CLOSE',index_col='Date',parse_dates=True)
cisco = pd.read_csv('CISCO_CLOSE',index_col='Date',parse_dates=True)
ibm = pd.read_csv('IBM_CLOSE',index_col='Date',parse_dates=True)
amzn = pd.read_csv('AMZN_CLOSE',index_col='Date',parse_dates=True)

stocks = pd.concat([aapl,cisco,ibm,amzn],axis=1)
stocks.columns = ['aapl','cisco','ibm','amzn']
stocks.head()

##### mean_daily returns
mean_daily_ret = stocks.pct_change(1).mean()
mean_daily_ret

##### correlation_daily returns
stocks.pct_change(1).corr()


##### simulating possible allocations
stocks.head()
stock_normed = stocks/stocks.iloc[0]       ## normalizing and plotting
stock_normed.plot()
stock_daily_ret = stocks.pct_change(1)     ## daily returns
stock_daily_ret.head()

#### log returns: detrending/normalizing
log_ret = np.log(stocks/stocks.shift(1))
log_ret.head()
log_ret.hist(bins=100,figsize=(12,6));     ## plotting each log_returns
plt.tight_layout()
log_ret.describe().transpose()  ## showing descriptive stats and transpose

## annualized mean_log_returns
log_ret.mean() * 252

# Compute pairwise covariance of columns
log_ret.cov()
log_ret.cov()*252 # annualized cov: multiply by days   












