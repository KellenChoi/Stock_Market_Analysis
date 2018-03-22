import pandas as pd
import quandl

##### Create a portfolio: techies(aapl. cisco, ibm, amzn)
start = pd.to_datetime('2012-01-01')
end = pd.to_datetime('2017-01-01')

##### getting data for the portfolio
aapl = quandl.get('WIKI/AAPL.11',start_date=start,end_date=end)
cisco = quandl.get('WIKI/CSCO.11',start_date=start,end_date=end)
ibm = quandl.get('WIKI/IBM.11',start_date=start,end_date=end)
amzn = quandl.get('WIKI/AMZN.11',start_date=start,end_date=end)

# Alternative: using csv files
# aapl = pd.read_csv('AAPL_CLOSE',index_col='Date',parse_dates=True)
# cisco = pd.read_csv('CISCO_CLOSE',index_col='Date',parse_dates=True)
# ibm = pd.read_csv('IBM_CLOSE',index_col='Date',parse_dates=True)
# amzn = pd.read_csv('AMZN_CLOSE',index_col='Date',parse_dates=True)
aapl.to_csv('AAPL_CLOSE')
cisco.to_csv('CISCO_CLOSE')
ibm.to_csv('IBM_CLOSE')
amzn.to_csv('AMZN_CLOSE')

##### Normalize prices ---> the same as cumulative daily returns
# Example
aapl.iloc[0]['Adj. Close']
for stock_df in (aapl,cisco,ibm,amzn):
    stock_df['Normed Return'] = stock_df['Adj. Close']/stock_df.iloc[0]['Adj. Close']
aapl.head()
aapl.tail()

##### Allocation: 3:2:4:1
for stock_df,allo in zip([aapl,cisco,ibm,amzn],[.3,.2,.4,.1]):
    stock_df['Allocation'] = stock_df['Normed Return']*allo
aapl.head()

##### Investment: $1,000,000 in the portfolio
for stock_df in [aapl,cisco,ibm,amzn]:
    stock_df['Position Values'] = stock_df['Allocation']*1000000

##### Total portfolio value
portfolio_val = pd.concat([aapl['Position Values'],cisco['Position Values'],ibm['Position Values'],amzn['Position Values']],axis=1)
portfolio_val.columns = ['AAPL Pos','CISCO Pos','IBM Pos','AMZN Pos']
portfolio_val.head()    
portfolio_val['Total Pos'] = portfolio_val.sum(axis=1)    ### adding total position
portfolio_val.head()   

##### Plotting 
import matplotlib.pyplot as plt
%matplotlib inline

### plotting the total portfolio values
portfolio_val['Total Pos'].plot(figsize=(10,8))
plt.title('Total Portfolio Value')

### plotting individual position values separately
portfolio_val.drop('Total Pos',axis=1).plot(kind='line')

portfolio_val.tail()  


##### Portfolio statistics
#### Returns
### Daily Returns of the portfolio
portfolio_val['Daily Return'] = portfolio_val['Total Pos'].pct_change(1)

### Cumulative Return
cum_ret = 100 * (portfolio_val['Total Pos'][-1]/portfolio_val['Total Pos'][0] -1 )
print('Our return {} was percent!'.format(cum_ret))

### Average Daily Return
portfolio_val['Daily Return'].mean()

#### Volatility
### Std Daily Return
portfolio_val['Daily Return'].std()

###### Shape Ratio: risk-adjusted return
SR = portfolio_val['Daily Return'].mean()/portfolio_val['Daily Return'].std()  #### Risk free = 0%
SR
Annualized_SR = (252**0.5)*SR
Annualized_SR

##### Plotting 
portfolio_val['Daily Return'].plot('kde')

### for each stocks 
aapl['Adj. Close'].pct_change(1).plot('kde')
ibm['Adj. Close'].pct_change(1).plot('kde')
amzn['Adj. Close'].pct_change(1).plot('kde')
cisco['Adj. Close'].pct_change(1).plot('kde')

#### annualized SR by using nympy
import numpy as np
np.sqrt(252)* (np.mean(.001-0.0002)/.001)
























