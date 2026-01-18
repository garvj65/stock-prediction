# 📈 Stock Price Prediction & Analysis using Machine Learning

This project implements an end-to-end stock analysis and price prediction system using historical market data and machine learning techniques. It is designed as an educational project to demonstrate data analysis, feature engineering, and regression-based modeling on financial time-series data.

---

## 🔍 Overview

The application:
- Fetches real historical stock data using Yahoo Finance
- Performs exploratory data analysis and visualization
- Computes technical indicators such as moving averages and daily returns
- Trains a machine learning model to estimate stock closing prices
- Evaluates model performance using multiple regression metrics
- Displays results through detailed plots and a performance dashboard

---

## 🧠 Concepts Covered

- Python data analysis (`pandas`, `numpy`)
- Stock market fundamentals (OHLCV, trends, volatility)
- Technical indicators (Moving Averages, Returns)
- Machine Learning (Linear Regression)
- Time-series aware train-test splitting
- Model evaluation (RMSE, R² Score, MAPE)
- Data visualization (`matplotlib`)

---

## 🛠️ Tech Stack

- **Python**
- **yfinance** – Stock market data
- **pandas & numpy** – Data manipulation
- **matplotlib** – Visualization
- **scikit-learn** – Machine learning models & metrics

---

## 📊 Features Used for Prediction

- **10-Day Moving Average (MA_10)** – Short-term trend
- **50-Day Moving Average (MA_50)** – Medium-term trend
- **Daily Return** – Price momentum and volatility

**Target Variable:** Closing Price

---

## 🤖 Model

- **Algorithm:** Linear Regression  
- **Train-Test Split:** 80% training, 20% testing (time-series preserved)  
- **Evaluation Metrics:**
  - RMSE (Root Mean Squared Error)
  - R² Score
  - MAPE (Mean Absolute Percentage Error)

---

## ⚠️ Disclaimer

This project is intended for **educational purposes only**.  
Stock prices are influenced by numerous external factors such as news, market sentiment, and macroeconomic events. The model captures historical patterns but does not guarantee future performance.

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-price-prediction.git
   cd stock-price-prediction
