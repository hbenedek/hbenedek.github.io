---
title: "Markowitz Portfolio Optimization"
layout: post
---

<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 200%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
         chtml: {
            scale: 1.3
        },
        svg: {
            scale: 1.3
        },
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>

Under construction.

TODO:
 - Loading the data
 - risk and return
 - calculating annualized values
 - quadratic programming 
 - solving the problem and plotting with random portfolios

---

First we load the data from yahoo, we use yfinance to do so. In tickers we specify which stocks to load, start and end simply denotes the time window we are interested in, in our case we chose a twelwe-year period between 2008 and 2020. The parameter interval is set to 1m , this will load one data point in each month (the first business day of each month), for twelve years it is 156 data points per stock. We are only interested in the closing price after adjustments so we take the columns Adjusted Close from our dataframe.

```python
import yfinance as yf
tickers = ['aapl', 'jnj', 'jpm', 'pg', 'xom', 'pfe', 'msft', 't', 'c', 'orcl', 'ge', 'wfc']
df = yf.download(tickers, data_source='yahoo', start='2008-01-01', end='2020-12-31', interval='1mo')['Adj Close'].dropna()
stocks = [t.upper() for t in tickers]

```

We can calculate the monthly realized returns with the formula $\huge{R_{t+1}=\frac{P_{t+1}}{P_{t}}} - 1. If we invest in a stock at time t and the quantity $\huge{R_{t+1}}$ is bigger than 0, then we made a profit and the return on our investment is $\huge{R_{t+1}}$ percent. Now we can calculate the realized annual return on an invetemnt, which is 
<center>
 $$\huge{1+R_{annual}=\prod_{t=1}^{12}(1+R_t)}$$
</center>
where $\huge{t}$ indicates the months in a year from January to December. We can compare the annual returns to see which investments performed better in a given year.

```python
returns = df / df.shift(1) - 1
```

If we look at the returns of *1* and *2* we see that the returns of *1* is not that spread out as the return on *2*, which we could win more if investing in *2*, but there is a bigger chance of getting negative returns as well. This property of an investment called the risk and measured by the standard deviation of the underlying random variable. We again use the empirical distribution of the realized returns to estimate the variance 
<center>
 $$\huge{Var(R)=\frac{1}{T-1}\sum_{t=1}^T (R_t-\overline{R})^2}$$
</center>
where $\huge{\overline{R}}$ denotes the average realized return.
