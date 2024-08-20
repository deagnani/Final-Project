# **Forex Trading Robot**

This project is a Forex Trading Robot that leverages various machine learning models, statistical methods, and trading strategies to make informed trading decisions in the Forex market. The robot is designed to analyze market data, predict price movements, and execute trades with a high probability of success.

## **Features**

### **1. Technical Indicators**
The robot calculates multiple technical indicators, including:
- Simple Moving Averages (SMA)
- Exponential Moving Averages (EMA)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Stochastic Oscillator
- Bollinger Bands
- Average True Range (ATR)
- Average Directional Index (ADX)
- Commodity Channel Index (CCI)
- Williams %R

### **2. Trading Strategies**
The robot employs several trading strategies, including:
- **Fibonacci Retracement**: Identifies potential reversal levels using Fibonacci retracement.
- **Breakout Strategy**: Detects breakouts when the price moves beyond Bollinger Bands with significant volume.
- **Mean Reversion**: Executes trades when the price deviates significantly from the mean (SMA50).
- **Volume Analysis**: Uses volume data to confirm the strength of a trend.
- **Trend Following**: Trades in the direction of the trend, using SMA50 and SMA200 as indicators.
- **Momentum Trading**: Leverages RSI and MACD to trade based on momentum.

### **3. Machine Learning Models**
The robot uses various machine learning models to predict market movements and estimate the probability of success:
- **XGBoost**: For classification of price movements.
- **Random Forest and Gradient Boosting**: For ensemble learning.
- **Naive Bayes**: For probabilistic predictions.
- **Monte Carlo Dropout**: To estimate uncertainty in predictions.
- **Quantile Regression**: For predicting the range of possible outcomes.

### **4. Reinforcement Learning**
A custom Gym environment is used to train a Reinforcement Learning (RL) model with PPO (Proximal Policy Optimization) to learn trading policies through interaction with the market.

### **5. Risk Management**
The robot includes advanced risk management features such as:
- **Value at Risk (VaR)**
- **Conditional Value at Risk (CVaR)**
- **Dynamic Position Sizing**: Adjusts position sizes based on account equity, volatility, and risk tolerance.

### **6. Portfolio Optimization**
The robot uses the Efficient Frontier algorithm to optimize the portfolio, maximizing the Sharpe ratio based on historical returns.

### **7. Sentiment Analysis**
The robot analyzes news headlines and other textual data to gauge market sentiment, influencing trading decisions.

## **Setup**

### **1. Prerequisites**
Ensure that you have Python 3.7+ installed. The following libraries are required:
- `MetaTrader5`
- `pandas`
- `numpy`
- `ta`
- `scikit-learn`
- `xgboost`
- `tensorflow`
- `statsmodels`
- `scikit-optimize`
- `gym`
- `stable-baselines3`
- `transformers`
- `vaderSentiment`
- `pypfopt`
- `scipy`

### **2. Installation**
Install the required Python packages using pip:
```bash
pip install MetaTrader5 pandas numpy ta scikit-learn xgboost tensorflow statsmodels scikit-optimize gym stable-baselines3 transformers vaderSentiment pypfopt scipy
