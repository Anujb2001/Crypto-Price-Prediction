# Cryptocurrency Price Prediction

## Overview

This project aims to predict cryptocurrency prices using various machine learning models. The goal is to develop an efficient and accurate prediction system that helps users make informed decisions in the volatile cryptocurrency market.
### Blog
https://anuj-bhardwaj.medium.com/machine-learning-models-for-crypto-price-prediction-2a4d30c01aa6

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Data](#data)
- [Technologies](#technologies)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Application UI](#ui)
  

## Features
- Predicts future prices of cryptocurrencies (e.g., Bitcoin, Ethereum) based on historical data.
- Implements various machine learning models for prediction.
- Supports time series analysis with advanced feature engineering.
- Visualizes predictions and market trends.
- Compares the performance of different models to find the best-performing one.

## Data
The dataset includes historical price data for various cryptocurrencies sourced from [API provider] or public datasets such as [Kaggle](https://www.kaggle.com/) or [CoinGecko](https://www.coingecko.com/).

Features include:
- **Date**: Date of price observation.
- **Open**: Opening price of the cryptocurrency.
- **High**: Highest price of the day.
- **Low**: Lowest price of the day.
- **Close**: Closing price of the cryptocurrency.
- **Volume**: Volume of trade during the day.
- **Market Cap**: Market capitalization of the cryptocurrency.

## Technologies
- **Python** (3.x)
- **Pandas**: Data manipulation and analysis
- **Numpy**: Numerical computing
- **Scikit-learn**: Machine learning models
- **TensorFlow/Keras**: Deep learning framework (if applicable)
- **Matplotlib/Seaborn**: Data visualization
- **Prophet**: Time series forecasting (optional)
- **API**: Cryptocurrency data fetching (e.g., from CoinGecko, Binance)

## Models
The following models are implemented and compared for performance:
- **Linear Regression**
- **Random Forest**
- **XGBoost**
- **LSTM (Long Short-Term Memory)** for time series analysis
- **ARIMA** for time series forecasting (if applicable)

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/Anujb2001/crypto-price-prediction.git
cd crypto-price-prediction
pip install -r requirements.txt
```
## Usage
```
streamlit run main.py
```
## UI
![image](https://github.com/user-attachments/assets/f8b9afcc-86a4-4aaf-a2b5-b4177ac9ebf7)


