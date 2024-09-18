# Task 4.	EDA Perform Exploratory Data Analysis for each selected cryptocurrency. explore temporal structure, visualize the distribution of observation, and investigate the change in distribution over intervals

import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas import DataFrame
from pandas import concat
import matplotlib.pyplot as plt

def plot_seasonal_decompose(df):
    # Decomposition
    result_add = seasonal_decompose(df['Close'], model ='addaptive', extrapolate_trend='freq', period=30)
    plt.rcParams.update({'figure.figsize': (5,5)})
    fig = result_add.plot()
    return fig

def plot_line_chart_price_over_time(coins_df_list):
    # Get all four coins data
    fig = plt.figure(figsize=(10, 6))
    for coin_name, df in coins_df_list.items():
        df['Close'].plot(label=coin_name)
    plt.title('Cryptocurrency Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.legend()
    return fig

# Distribution of Cryptocurrency Prices
def plot_distribution_of_coin_prices(coins_df_list):
    # Get all four coins data
    fig = plt.figure(figsize=(8, 6))
    for coin_name, df in coins_df_list.items():
        sns.kdeplot(df['Close'], label=coin_name)
    plt.title('Distribution of Cryptocurrency Prices')
    plt.xlabel('Price (USD)')
    plt.ylabel('Density (%)')
    plt.grid(True)
    plt.legend()
    return fig

def box_plot_monthly_distrb(df, coin_name):
    # Investigating Change in Distribution Over Intervals (e.g., Monthly)
    # Define month names in sorted order
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Convert index to datetime if it's not already
    df.index = pd.to_datetime(df.index)
    df['Month'] = df.index.strftime('%b')
    # Convert Month column to categorical data type with defined order
    df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Month', y='Close')
    plt.title(f'Monthly Distribution of {coin_name} coin Prices')
    plt.xlabel('Month')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.show()
    return fig

def plot_time_series_autocorrelation(df, coin_name):
    plt.rcParams.update({'figure.figsize' : (15,9), 'figure.dpi' : 120})
    # The Genuine Series
    fig, axes = plt.subplots(3, 2)
    axes[0, 0].plot(df["Close"]); axes[0, 0].set_title('The Genuine Series')
    plot_acf(df['Close'], ax = axes[0, 1])

    # Order of Differencing: First
    diff1=df["Close"].diff().dropna()
    axes[1, 0].plot(diff1); axes[1, 0].set_title('Order of Differencing: First')
    plot_acf(diff1, ax = axes[1, 1])

    # Order of Differencing: Second
    diff2 = df["Close"].diff().diff().dropna()
    axes[2, 0].plot(diff2); axes[2, 0].set_title('Order of Differencing: Second')
    plot_acf(diff2, ax = axes[2, 1])
    plt.suptitle(f'{coin_name} Coin: Time Series & Autocorrelation')
    plt.tight_layout()
    plt.show()
    return fig

def scatter_plot_lag (df, coin_name):
    # Assuming df is your DataFrame
    series = df["Close"]
    values = DataFrame(series.values)
    lags = 7
    columns = [values]
    for i in range(1, (lags + 1)):
        columns.append(values.shift(i))
    dataframe = concat(columns, axis=1)
    columns = ['t+1']
    for i in range(1, (lags + 1)):
        columns.append('t-' + str(i))
    dataframe.columns = columns

    # Increase figure size and add title, xlabel, ylabel
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(f'{coin_name} Coin Scatter Plot of t+1 vs t-lag')
    for i in range(1, (lags + 1)):
        ax = plt.subplot(240 + i)
        ax.set_title('t+1 vs t-' + str(i))
        ax.scatter(x=dataframe['t+1'].values, y=dataframe['t-'+str(i)].values)
        ax.set_xlabel('t+1')
        ax.set_ylabel('t-' + str(i))

    plt.tight_layout()
    return fig


def plot_moving_average(df, coin_name):
    window_size = 30
    rolling_avg = df['Close'].rolling(window=window_size).mean()
    fig = plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label='Close price')
    plt.plot(rolling_avg.index, rolling_avg, label=f'{window_size}-Day Moving Average', color='red')
    plt.title(f'Moving Average of {coin_name} coin Close price')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    return fig

