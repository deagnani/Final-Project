import pandas as pd
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import MetaTrader5 as mt5
import xgboost as xgb
import tensorflow as tf
import statsmodels.api as sm
from skopt import BayesSearchCV
import gym
from stable_baselines3 import PPO
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

# Initialize MetaTrader 5
def initialize_mt5():
    if not mt5.initialize():
        print(f"MetaTrader 5 initialization failed, error code: {mt5.last_error()}")
        return False
    return True

# Login to MetaTrader 5 account
def login_mt5(account, password, server):
    authorized = mt5.login(account, password, server)
    if not authorized:
        print(f"Login failed, error code: {mt5.last_error()}")
        return False
    print(f"Connected to account #{account}")
    return True

# Fetch historical data for a symbol
def fetch_data(symbol, timeframe, n_bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        print(f"Failed to fetch data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Calculate indicators
def calculate_indicators(df):
    df['SMA50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['EMA50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['SMA200'] = ta.trend.sma_indicator(df['close'], window=200)
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['close'])
    df['Stochastic'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    df['Bollinger_High'] = ta.volatility.bollinger_hband(df['close'])
    df['Bollinger_Low'] = ta.volatility.bollinger_lband(df['close'])
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'])
    df['CCI'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
    df['Williams_R'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
    df['Tick_Volume'] = df['tick_volume']
    df['Returns'] = df['close'].pct_change()
    df['Log_Returns'] = np.log1p(df['close'].pct_change())
    df['Price_Range'] = df['high'] - df['low']
    df['Volatility'] = df['Price_Range'] / df['close']
    df.dropna(inplace=True)
    return df

# Calculate Fibonacci retracement levels
def calculate_fibonacci_levels(df):
    high_price = df['high'].max()
    low_price = df['low'].min()
    diff = high_price - low_price
    levels = {
        'level_0': high_price,
        'level_23.6': high_price - 0.236 * diff,
        'level_38.2': high_price - 0.382 * diff,
        'level_50.0': high_price - 0.5 * diff,
        'level_61.8': high_price - 0.618 * diff,
        'level_78.6': high_price - 0.786 * diff,
        'level_100': low_price,
    }
    return levels

# Define strategies
def define_strategies(df, fib_levels):
    df['signal'] = 0
    df['strategy'] = ''
    for i in range(1, len(df)):
        # Fibonacci retracement strategy
        close = df['close'].iloc[i]
        if close <= fib_levels['level_61.8'] and close >= fib_levels['level_50.0']:
            df.loc[i, 'signal'] = 1  # Buy signal in the golden zone
            df.loc[i, 'strategy'] = 'Fibonacci Retracement'
        elif close >= fib_levels['level_38.2'] and close <= fib_levels['level_50.0']:
            df.loc[i, 'signal'] = -1  # Sell signal above the golden zone
            df.loc[i, 'strategy'] = 'Fibonacci Retracement'
        # Breakout strategy
        elif (df['close'].iloc[i] > df['Bollinger_High'].iloc[i]) and (df['Tick_Volume'].iloc[i] > df['Tick_Volume'].mean()):
            df.loc[i, 'signal'] = 1
            df.loc[i, 'strategy'] = 'Breakout'
        # Mean reversion strategy
        elif (df['close'].iloc[i] < df['SMA50'].iloc[i]) and (df['RSI'].iloc[i] > 70):
            df.loc[i, 'signal'] = -1
            df.loc[i, 'strategy'] = 'Mean Reversion'
        # Volume analysis strategy
        elif (df['close'].iloc[i] < df['Bollinger_Low'].iloc[i]) and (df['Tick_Volume'].iloc[i] > df['Tick_Volume'].mean()):
            df.loc[i, 'signal'] = -1
            df.loc[i, 'strategy'] = 'Volume Analysis'
        # Trend following strategy
        elif (df['SMA50'].iloc[i] > df['SMA200'].iloc[i]) and (df['ADX'].iloc[i] > 20):
            df.loc[i, 'signal'] = 1  # Buy signal for uptrend
            df.loc[i, 'strategy'] = 'Trend Following'
        elif (df['SMA50'].iloc[i] < df['SMA200'].iloc[i]) and (df['ADX'].iloc[i] > 20):
            df.loc[i, 'signal'] = -1  # Sell signal for downtrend
            df.loc[i, 'strategy'] = 'Trend Following'
        # Momentum trading strategy
        elif (df['RSI'].iloc[i] > 70) and (df['MACD'].iloc[i] > 0):
            df.loc[i, 'signal'] = 1  # Buy signal for strong upward momentum
            df.loc[i, 'strategy'] = 'Momentum Trading'
        elif (df['RSI'].iloc[i] < 30) and (df['MACD'].iloc[i] < 0):
            df.loc[i, 'signal'] = -1  # Sell signal for strong downward momentum
            df.loc[i, 'strategy'] = 'Momentum Trading'
    return df

# Monte Carlo simulation
def monte_carlo_simulation(model, features, n_simulations=1000):
    positive_predictions = 0
    for _ in range(n_simulations):
        if model.predict(features)[0] == 1:
            positive_predictions += 1
    return positive_predictions / n_simulations

# Enhanced Bayesian update
def bayesian_update(prior, likelihood):
    if likelihood == 0:
        return 0
    return (prior * likelihood) / ((prior * likelihood) + ((1 - prior) * (1 - likelihood)))

# Additional Probability Method: Gaussian Naive Bayes
def naive_bayes_probability(model, features):
    proba = model.predict_proba(features)[0][1]  # Probability of the positive class
    return proba

# Calibration methods
def calibrate_model(model, X_train, y_train, method='sigmoid'):
    calibrated_model = CalibratedClassifierCV(base_estimator=model, method=method, cv='prefit')
    calibrated_model.fit(X_train, y_train)
    return calibrated_model

# Bayesian Optimization
def bayesian_optimization(model, param_grid, X_train, y_train):
    opt = BayesSearchCV(model, param_grid, n_iter=32, cv=3, random_state=42, n_jobs=-1)
    opt.fit(X_train, y_train)
    return opt.best_estimator_

# Monte Carlo Dropout model
def create_monte_carlo_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_monte_carlo_model(X_train, y_train, X_test, y_test):
    input_shape = X_train.shape[1]
    model = create_monte_carlo_model(input_shape)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    return model

def monte_carlo_predictions(model, X, n_iter=100):
    predictions = [model(X, training=True) for _ in range(n_iter)]
    predictions = np.array(predictions)
    mean_predictions = predictions.mean(axis=0)
    std_predictions = predictions.std(axis=0)
    return mean_predictions, std_predictions

# Quantile Regression
def quantile_regression(X_train, y_train, quantile=0.5):
    model = sm.QuantReg(y_train, sm.add_constant(X_train)).fit(q=quantile)
    return model

# Place a buy order
def place_buy_order(symbol, volume, sl, tp):
    price = mt5.symbol_info_tick(symbol).ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Buy order failed for {symbol}: {result.comment}")
    else:
        print(f"Buy order placed for {symbol} at price {price}")
    return result

# Place a sell order
def place_sell_order(symbol, volume, sl, tp):
    price = mt5.symbol_info_tick(symbol).bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Sell order failed for {symbol}: {result.comment}")
    else:
        print(f"Sell order placed for {symbol} at price {price}")
    return result

# Market sentiment analysis
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']

# Fetch and analyze news sentiment
def fetch_news_sentiment():
    # Placeholder function to simulate fetching news headlines
    headlines = [
        "Global stocks rally as economic optimism returns",
        "Oil prices drop amid fears of oversupply",
        "Tech stocks soar on strong earnings reports",
        "Central bank signals rate hike to combat inflation"
    ]
    sentiments = [analyze_sentiment(headline) for headline in headlines]
    return np.mean(sentiments)  # Average sentiment score

# Portfolio optimization
def optimize_portfolio(df_dict):
    returns_dict = {}
    for symbol, df in df_dict.items():
        returns_dict[symbol] = df['Returns']
    returns_df = pd.DataFrame(returns_dict)
    mu = expected_returns.mean_historical_return(returns_df)
    S = risk_models.sample_cov(returns_df)
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    return cleaned_weights

# Value at Risk (VaR) function
def calculate_var(df, confidence_level=0.95):
    returns = df['Returns']
    mean = np.mean(returns)
    std_dev = np.std(returns)
    var = norm.ppf(1 - confidence_level, mean, std_dev)
    return var

# Conditional Value at Risk (CVaR) function
def calculate_cvar(df, confidence_level=0.95):
    returns = df['Returns']
    var = calculate_var(df, confidence_level)
    cvar = np.mean([x for x in returns if x <= var])
    return cvar

# Dynamic position sizing
def calculate_position_size(balance, risk_per_trade, sl_price, entry_price):
    risk_amount = balance * risk_per_trade
    position_size = risk_amount / abs(entry_price - sl_price)
    return position_size

# Custom Gym environment for trading
class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(df.columns),), dtype=np.float16)

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.df.iloc[self.current_step].values
        return obs

    def step(self, action):
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            self.current_step = 0

        current_price = self.df.loc[self.current_step, 'close']

        if action == 1:  # Buy
            shares_bought = self.balance // current_price
            self.balance -= shares_bought * current_price
            self.shares_held += shares_bought

        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price
            self.total_shares_sold += self.shares_held
            self.total_sales_value += self.shares_held * current_price
            self.shares_held = 0

        reward = self.balance + self.shares_held * current_price

        done = self.current_step == len(self.df) - 1

        return self._next_observation(), reward, done, {}

# Train the RL model
def train_rl_model(df):
    env = TradingEnv(df)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("trading_model")
    return model

# Define the trading logic using strategies and machine learning model
def trading_logic(df, model, nb_model, scaler, symbol, balance, fib_levels):
    df['prediction'] = 0  # Add a column for predictions
    df['entry_price'] = 0.0  # Add a column for entry prices, ensuring float type
    df['sl'] = 0.0  # Add a column for stop-loss, ensuring float type
    df['tp'] = 0.0  # Add a column for take-profit, ensuring float type

    max_trades = 4  # Set the maximum number of trades per pair
    trade_count = 0
    min_lot_size = 0.01  # Minimum lot size, adjust as per broker requirements
    volume_step = 0.01  # Volume step, adjust as per broker requirements

    # Calculate the trade volume based on the balance
    trade_volume = (balance * 0.1) / df['close'].iloc[-1]
    if pd.isna(trade_volume) or trade_volume <= 0:
        trade_volume = min_lot_size
    trade_volume = max(min_lot_size, round(trade_volume / volume_step) * volume_step)

    prior_probability = 0.5  # Initial prior probability (neutral)

    for i in range(1, len(df)):
        if trade_count >= max_trades:
            break

        feature_names = ['SMA50', 'EMA50', 'RSI', 'MACD', 'Stochastic', 'Bollinger_High', 'Bollinger_Low', 'ATR', 
                         'ADX', 'CCI', 'Williams_R', 'Tick_Volume', 'Returns', 'Log_Returns', 'Price_Range', 'Volatility', 'signal']
        features = pd.DataFrame([df[feature_names].iloc[i].values], columns=feature_names)
        features_scaled = scaler.transform(features)

        # Predict using the model
        prediction = model.predict(features_scaled)[0]
        
        # Monte Carlo simulation to estimate probability
        monte_carlo_probability = monte_carlo_simulation(model, features_scaled)

        # Bayesian update
        updated_probability = bayesian_update(prior_probability, monte_carlo_probability)

        # Naive Bayes probability
        nb_probability = naive_bayes_probability(nb_model, features_scaled)

        # Final combined probability
        final_probability = (updated_probability + nb_probability) / 2

        # Determine the prediction
        df.loc[i, 'prediction'] = prediction  # Store the prediction

        # Calculate SL and TP with a 1:3 ratio
        if prediction == 1:  # Buy signal
            sl = df['close'].iloc[i] * 0.995
            tp = df['close'].iloc[i] * (1 + 3 * (1 - 0.995))
        else:  # Sell signal
            sl = df['close'].iloc[i] * 1.005
            tp = df['close'].iloc[i] * (1 - 3 * (0.005))

        print(f"Index: {i}, Prediction: {prediction}, Final Probability: {final_probability:.2f}")
        print(f"Conditions: Close: {df['close'].iloc[i]}, SMA50: {df['SMA50'].iloc[i]}, RSI: {df['RSI'].iloc[i]}, Tick Volume: {df['Tick_Volume'].iloc[i]}, Tick Volume Mean: {df['Tick_Volume'].mean()}")

        # Logging the conditions and decisions
        if prediction == 1 and final_probability >= 0.8:
            # Place a buy order if the conditions are met
            df.loc[i, 'entry_price'] = df['close'].iloc[i]
            df.loc[i, 'sl'] = sl
            df.loc[i, 'tp'] = tp
            result = place_buy_order(symbol, trade_volume, sl, tp)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Buy order executed: {symbol}, Volume: {trade_volume}, SL: {sl}, TP: {tp}")
                trade_count += 1
            else:
                print(f"Buy order failed: {result.comment}")

        elif prediction == -1 and final_probability >= 0.8:
            # Place a sell order if the conditions are met
            df.loc[i, 'entry_price'] = df['close'].iloc[i]
            df.loc[i, 'sl'] = sl
            df.loc[i, 'tp'] = tp
            result = place_sell_order(symbol, trade_volume, sl, tp)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Sell order executed: {symbol}, Volume: {trade_volume}, SL: {sl}, TP: {tp}")
                trade_count += 1
            else:
                print(f"Sell order failed: {result.comment}")

    return df

# Main function
def main():
    # Initialize MetaTrader 5
    if not initialize_mt5():
        return  # Exit if initialization fails

    # Account details
    login = 1520331734
    password = "?VS69R=xKs$q6p"
    server = "FTMO-Demo2"

    # Attempt to login
    if not login_mt5(login, password, server):
        mt5.shutdown()
        return  # Exit if login fails

    # Get account information
    account_info = mt5.account_info()
    if account_info is None:
        print(f"Failed to retrieve account info, error code = {mt5.last_error()}")
        mt5.shutdown()
        return  # Exit if account info retrieval fails

    print(account_info)

    # Define symbols and parameters
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'XAUUSD', 'XAGUSD', 'WTI', 'BRENT', 'DAX30', 'SPX500']
    timeframe = mt5.TIMEFRAME_H1
    n_bars = 500  # Increase the number of bars to ensure sufficient data

    # Fetch news sentiment
    news_sentiment = fetch_news_sentiment()
    print(f"News Sentiment: {news_sentiment}")

    df_dict = {}

    for symbol in symbols:
        df = fetch_data(symbol, timeframe, n_bars)
        if df is None or df.empty:
            print(f"Skipping {symbol} due to fetch_data error.")
            continue
        df = calculate_indicators(df)
        if df.empty:
            print(f"No data available after indicator calculation for {symbol}")
            continue

        fib_levels = calculate_fibonacci_levels(df)
        df = define_strategies(df, fib_levels)
        df_dict[symbol] = df

    # Optimize portfolio
    weights = optimize_portfolio(df_dict)
    print("Optimized Portfolio Weights:", weights)

    for symbol, df in df_dict.items():
        features = df[['SMA50', 'EMA50', 'RSI', 'MACD', 'Stochastic', 'Bollinger_High', 'Bollinger_Low', 'ATR', 
                       'ADX', 'CCI', 'Williams_R', 'Tick_Volume', 'Returns', 'Log_Returns', 'Price_Range', 'Volatility', 'signal']]
        target = (df['close'].shift(-1) > df['close']).astype(int)  # Target: 1 if price goes up, 0 if price goes down

        features = features.dropna()
        target = target.dropna()

        # Align features and target
        aligned_features, aligned_target = features.align(target, join='inner', axis=0)

        if aligned_features.empty or aligned_target.empty:
            print("No data to train the model.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(aligned_features, aligned_target, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the models
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        
        param_grid = {
            'n_estimators': (50, 200),
            'max_depth': (3, 7),
            'learning_rate': (0.01, 0.2)
        }
        
        # Bayesian Optimization for hyperparameter tuning
        optimized_xgb_model = bayesian_optimization(xgb_model, param_grid, X_train, y_train)

        y_pred = optimized_xgb_model.predict(X_test)
        print(f"Optimized ROC AUC Score: {roc_auc_score(y_test, y_pred)}")
        print(f"Optimized F1 Score: {f1_score(y_test, y_pred)}")

        # Calibrate the optimized model
        calibrated_xgb_model = calibrate_model(optimized_xgb_model, X_train, y_train, method='sigmoid')

        # Make predictions with calibrated model
        y_pred_calibrated = calibrated_xgb_model.predict_proba(X_test)[:, 1]
        print(f"Calibrated ROC AUC Score: {roc_auc_score(y_test, y_pred_calibrated)}")
        print(f"Calibrated F1 Score: {f1_score(y_test, calibrated_xgb_model.predict(X_test))}")

        # Train a Naive Bayes model
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)

        # Train Random Forest and Gradient Boosting models
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        gbm_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gbm_model.fit(X_train, y_train)

        # Evaluate models
        for model in [optimized_xgb_model, calibrated_xgb_model, rf_model, gbm_model, nb_model]:
            y_pred = model.predict(X_test)
            print(f"Model: {type(model).__name__}")
            print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
            print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred)}")
            print(f"F1 Score: {f1_score(y_test, y_pred)}\n")

        # Train Monte Carlo Dropout model
        mc_model = train_monte_carlo_model(X_train, y_train, X_test, y_test)

        # Make Monte Carlo predictions
        mean_predictions, std_predictions = monte_carlo_predictions(mc_model, X_test)
        for mean, std in zip(mean_predictions, std_predictions):
            print(f"Monte Carlo Prediction: {mean}, Uncertainty: {std}")

        # Train Quantile Regression model
        qr_model = quantile_regression(X_train, y_train, quantile=0.5)
        qr_predictions = qr_model.predict(sm.add_constant(X_test))
        for pred in qr_predictions:
            print(f"Quantile Prediction: {pred}")

        # Apply trading logic with the best model (Calibrated XGBoost in this case)
        df = trading_logic(df, calibrated_xgb_model, nb_model, scaler, symbol, account_info.balance, fib_levels)
        print(df[['time', 'close', 'signal', 'strategy', 'prediction', 'entry_price', 'sl', 'tp']].tail())

    # Train the RL model
    for symbol, df in df_dict.items():
        rl_model = train_rl_model(df)
        env = TradingEnv(df)
        obs = env.reset()
        while True:
            action, _ = rl_model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            if done:
                obs = env.reset()

    # Shutdown MetaTrader 5 connection
    mt5.shutdown()

if __name__ == '__main__':
    main()
