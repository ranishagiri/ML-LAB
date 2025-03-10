import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define stock tickers
tickers = ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS"]

# Download stock data
data = yf.download(tickers, start="2024-01-01", end="2024-12-30", group_by='ticker')

# Create plots for each bank
plt.figure(figsize=(15, 12))

for i, ticker in enumerate(tickers):
    stock_data = data[ticker]

    # Calculate Daily Returns
    stock_data["Daily Return"] = stock_data["Close"].pct_change()

    # Closing Price Plot
    plt.subplot(len(tickers), 2, 2 * i + 1)
    plt.plot(stock_data.index, stock_data["Close"], label="Closing Price", color="blue")
    plt.title(f"{ticker} - Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(True)

    # Daily Returns Plot
    plt.subplot(len(tickers), 2, 2 * i + 2)
    plt.plot(stock_data.index, stock_data["Daily Return"], label="Daily Return", color="orange")
    plt.title(f"{ticker} - Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
