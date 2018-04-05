####  Model CAPM as a simple linear regression
from scipy import stats
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

# getting data from web
spy_etf = web.DataReader('SPY','morningstar')
aapl = web.DataReader('AAPL','morningstar')

# plotting closing prices
aapl['Close'].plot(label='AAPL',figsize=(10,8))
spy_etf['Close'].plot(label='SPY Index')
plt.legend()

# compare cumulative returns
aapl['Cumulative'] = aapl['Close']/aapl['Close'].iloc[0]
spy_etf['Cumulative'] = spy_etf['Close']/spy_etf['Close'].iloc[0]

# ploting cumulative returns
aapl['Cumulative'].plot(label='AAPL',figsize=(10,8))
spy_etf['Cumulative'].plot(label='SPY Index')
plt.legend()
plt.title('Cumulative Return')

# daily returns
aapl['Daily Return'] = aapl['Close'].pct_change(1)
spy_etf['Daily Return'] = spy_etf['Close'].pct_change(1)

# scatter plot to see relationship between index(spy) and appl
plt.scatter(aapl['Daily Return'],spy_etf['Daily Return'],alpha=0.3)

# histogram of daily returns: for volatility 
aapl['Daily Return'].hist(bins=100)
spy_etf['Daily Return'].hist(bins=100)

# regressing to get beta, alpha, r_value, p_value, std
beta,alpha,r_value,p_value,std_err = stats.linregress(aapl['Daily Return'].iloc[1:],spy_etf['Daily Return'].iloc[1:])

beta  ### 0.33110869773286694
alpha  ## 0.00013374331403944492
r_value


### What if our stock was completely related to SP500?
# creating noise (random, normal)
noise = np.random.normal(0,0.001,len(spy_etf['Daily Return'].iloc[1:]))
noise
# adding noise to the data
spy_etf['Daily Return'].iloc[1:] + noise
# regressing
beta,alpha,r_value,p_value,std_err = stats.linregress(spy_etf['Daily Return'].iloc[1:]+noise,spy_etf['Daily Return'].iloc[1:])
beta  ### 0.9859551151234642
alpha  ### 3.1960744063722134e-05
