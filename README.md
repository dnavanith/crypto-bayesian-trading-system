# 🚀 Cryptocurrency Bayesian Trading System

A comprehensive algorithmic trading system for cryptocurrencies using Gaussian Naive Bayes machine learning models with advanced technical analysis and real-time market data integration.

## 📊 Features

### Core Functionality
- **Real-time Data Collection**: Live mark price data from KuCoin Futures API
- **80+ Technical Indicators**: Comprehensive analysis including MACD variants, RSI periods, Stochastic oscillators, Williams %R, Bollinger Bands, and volatility metrics
- **Machine Learning**: Gaussian Naive Bayes model with rolling window training
- **Live Trading Simulation**: Position management with confidence-based decision making
- **No Look-Ahead Bias**: Proper backtesting methodology using only historical data

### Technical Implementation
- **Data Pipeline**: Automated collection of 180+ days of minute-level cryptocurrency data
- **Feature Engineering**: 80+ technical indicators with lagged features to prevent data leakage
- **Model Training**: Rolling window approach with 1, 2, 3, and 6-month training periods
- **Performance Metrics**: Comprehensive tracking including accuracy, win rate, profit factor, drawdown analysis
- **Cloud Deployment**: Google Cloud integration for 24/7 automated trading

## 🛠️ Technology Stack

- **Language**: Python 3.8+
- **Machine Learning**: scikit-learn (Gaussian Naive Bayes)
- **Technical Analysis**: TA-Lib, pandas, numpy
- **API Integration**: KuCoin Futures API
- **Cloud Platform**: Google Cloud Compute Engine
- **Session Management**: tmux for persistent cloud sessions
- **Data Processing**: pandas, numpy for large-scale data manipulation

## 📈 Key Achievements

- **Data Processing**: Successfully collected and processed 260,000+ minute-level data points
- **Feature Engineering**: Implemented 80+ technical indicators without look-ahead bias
- **Model Performance**: Achieved realistic backtesting results with proper temporal validation
- **Live Deployment**: Successfully deployed on Google Cloud for continuous operation
- **Risk Management**: Implemented comprehensive position sizing and stop-loss mechanisms


