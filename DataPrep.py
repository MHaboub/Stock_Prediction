import warnings
warnings.filterwarnings('ignore')

# PART 1: Stock Data Collection (Minimized)
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def get_stock_data(target_ticker, related_tickers, start_date='2020-01-01'):
    """Get stock data table with y (target) and x1-x9 (related tickers)"""
    print(f"📊 Fetching stock data for {target_ticker}...")
    print('ticker:', target_ticker)
    target_data = yf.download(target_ticker, start=start_date)["Close"]
    print(f"✅ Target data for {target_ticker} fetched: {target_data.shape}")
    print(target_data.head())
    # Fix: handle DataFrame or Series
    if isinstance(target_data, pd.DataFrame):
        if 'Close' in target_data.columns:
            y_series = target_data['Close']
        else:
            y_series = target_data.iloc[:, 0]
    else:
        y_series = target_data
    if isinstance(y_series, pd.Series) and not y_series.empty:
        df = pd.DataFrame({'y': y_series})
    else:
        raise ValueError(f"No data found for {target_ticker}. Please check the ticker symbol and date range.")
    
    # Get related tickers (limit to 9)
    for i, ticker in enumerate(related_tickers[:12], 1):
        try:
            ticker_data = yf.download(ticker, start=start_date)["Close"]
            print(f"✅ Data for {ticker} fetched: {ticker_data.shape}")
            print(ticker_data.head())
            # Handle DataFrame or Series, and check for all-NaN
            if isinstance(ticker_data, pd.DataFrame):
                if 'Close' in ticker_data.columns:
                    series = ticker_data['Close']
                else:
                    series = ticker_data.iloc[:, 0]
            else:
                series = ticker_data
            if isinstance(series, pd.Series) and not series.empty and not series.isna().all():
                df[f'x{i}'] = series.reindex(df.index, method='ffill')
            else:
                print(f"❌ No data for {ticker}")
        except Exception as e:
            print(f"❌ Failed to get {ticker}: {e}")
    
    df = df.dropna()
    print(f"✅ Stock data ready: {df.shape}")
    print(df.head())
    print(df.tail())
    return df

# PART 2: Sentiment Analysis (Minimized)
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
from dateutil import parser as date_parser
import urllib.parse

def get_sentiment_data(ticker, start_date='2020-01-01', end_date=None):
    """Generate sentiment analysis table using Google News headlines"""
    print("🎭 Generating sentiment data from Google News...")
    end_date = end_date or datetime.now().strftime('%Y-%m-%d')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    analyzer = SentimentIntensityAnalyzer()
    sentiment_data = []
    # Fetch Google News RSS feed for the ticker
    query = f"{ticker} stock"
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    # Build a date->headline mapping (use only the first headline per day)
    news_by_date = {}
    for entry in feed.entries:
        pub_date = date_parser.parse(entry.published).date()
        if pub_date not in news_by_date:
            news_by_date[pub_date] = entry.title
    for date in dates:
        headline = news_by_date.get(date.date(), f"No major news for {ticker} on {date.date()}")
        # TextBlob sentiment
        blob = TextBlob(headline)
        polarity = blob.sentiment.polarity
        # VADER sentiment
        vader_scores = analyzer.polarity_scores(headline)
        compound = vader_scores['compound']
        # Combined sentiment
        combined = (polarity * 0.4 + compound * 0.6)
        sentiment_data.append({
            'Date': date,
            'Polarity': polarity,
            'Compound': compound,
            'Combined_Sentiment': combined
        })
    sentiment_df = pd.DataFrame(sentiment_data)
    sentiment_df.set_index('Date', inplace=True)
    print(f"✅ Sentiment data ready: {sentiment_df.shape}")
    print(sentiment_df.head())
    print(sentiment_df.tail())
    return sentiment_df

# PART 3: LSTM Prediction (Minimized)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

import matplotlib.pyplot as plt

def predict_stock_lstm(stock_data, sentiment_data, prediction_days=30, time_steps=60):
    """Merge data, train LSTM, and predict future prices"""
    print("🔗 Merging data and training LSTM...")
    
    # Merge stock and sentiment data
    combined_data = pd.merge(stock_data, sentiment_data, left_index=True, right_index=True, how='inner')
    
    combined_data = combined_data.dropna()
    
    print(f"📊 Combined data shape: {combined_data.shape}")
    print(combined_data.head())
    
    # Prepare features and target
    feature_cols = [col for col in combined_data.columns if col != 'y']
    X = combined_data[feature_cols].values
    y = combined_data['y'].values.reshape(-1, 1)
    
    # Scale data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Create sequences
    X_seq, y_seq = [], []
    for i in range(time_steps, len(X_scaled)):
        X_seq.append(X_scaled[i-time_steps:i])
        y_seq.append(y_scaled[i])
    
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    
    # Split data
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    print(f"🚀 Training data: {X_train.shape}, Test data: {X_test.shape}")
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    print("🔥 Training LSTM model...")
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                       validation_data=(X_test, y_test), verbose=0)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_actual = scaler_y.inverse_transform(y_pred)
    y_test_actual = scaler_y.inverse_transform(y_test)
    
    # Calculate accuracy
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
    print(f"📊 Model RMSE: ${rmse:.2f}")
    print(f"📊 Model MAE: ${mae:.2f}")
    print(f"📊 Model R2: {r2:.4f}")
    print(f"📊 Model MAPE: {mape:.2f}%")
    
    # Predict future
    print(f"🔮 Predicting next {prediction_days} days...")
    last_sequence = X_scaled[-time_steps:]
    future_predictions = []
    
    for _ in range(prediction_days):
        next_pred = model.predict(last_sequence.reshape(1, time_steps, -1), verbose=0)
        future_predictions.append(next_pred[0, 0])
        
        # Update sequence (simplified)
        new_row = last_sequence[-1].copy()
        new_row[0] = next_pred[0, 0]  # Update first feature with prediction
        last_sequence = np.vstack([last_sequence[1:], new_row])
    
    # Convert predictions to actual prices
    future_prices = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    
    # Create future dates
    last_date = combined_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq='D')
    
    # Results DataFrame
    results = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': future_prices
    })
    
    current_price = combined_data['y'].iloc[-1]
    future_price = future_prices[-1]
    change_pct = ((future_price - current_price) / current_price) * 100
    
    print(f"✅ Prediction completed!")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   Price in {prediction_days} days: ${future_price:.2f}")
    print(f"   Expected Change: {change_pct:+.1f}%")
    
    return results, model, (scaler_X, scaler_y), {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2), 'mape': float(mape)}

# COMPLETE WORKFLOW FUNCTION
def run_stock_prediction(target_ticker, related_tickers, prediction_days=30):
    """Complete workflow: Get data → Analyze sentiment → Predict with LSTM"""
    print(f"🎯 Starting stock prediction for {target_ticker}")
    print("=" * 50)
    
    # Step 1: Get stock data
    stock_data = get_stock_data(target_ticker, related_tickers)
    
    # Step 2: Get sentiment data
    start_date = stock_data.index.min().strftime('%Y-%m-%d')
    end_date = stock_data.index.max().strftime('%Y-%m-%d')
    sentiment_data = get_sentiment_data(target_ticker, start_date, end_date)
    
    # Step 3: LSTM prediction
    predictions, model, scalers, metrics = predict_stock_lstm(stock_data, sentiment_data, prediction_days)
    
    print("\n🎉 PREDICTION COMPLETE!")
    print("=" * 50)
    return predictions, stock_data, sentiment_data, model, metrics

# EXAMPLE USAGE
if __name__ == "__main__":
    # Define parameters
    TARGET = "TSLA"
    #RELATED = ["AAPL", "GOOGL", "AMZN", "MSFT", "NVDA", "META", "NFLX", "AMD", "SPOT"]
    RELATED = [
    "NIO",   # NIO Inc.
    "LI",    # Li Auto Inc.
    "XPEV",  # XPeng Inc.
    "BYDDY", # BYD Company (OTC)
    "LCID",  # Lucid Group
    "RIVN",  # Rivian Automotive
    "FSR",   # Fisker Inc.
    "VFS",   # VinFast
    "CHPT",  # ChargePoint
    "BLNK",  # Blink Charging
    "EVGO"   # EVgo Inc.
]
    DAYS = 30
    
    # Run complete prediction
    predictions, stock_data, sentiment_data, model, metrics = run_stock_prediction(TARGET, RELATED, DAYS)
    
    # Display results
    print("\n📊 STOCK DATA SAMPLE:")
    print(stock_data.head())
    
    print("\n🎭 SENTIMENT DATA SAMPLE:")
    print(sentiment_data.head())
    
    print("\n🔮 FUTURE PREDICTIONS:")
    print(predictions.head(10))
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(predictions['Date'], predictions['Predicted_Price'], 'r-', linewidth=2)
    plt.title(f'{TARGET} Price Prediction - Next {DAYS} Days')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()