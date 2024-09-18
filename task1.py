# Task 1.	Get Data by using Yahoo Finance
import yfinance as yf
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# load the data if not downloaded
def get_year_data_by_coin_name(coin_name, field_name="Close"):
    if os.path.exists(f"coins/{coin_name}.csv"):
        df = pd.read_csv(f"coins/{coin_name}.csv", index_col=0, parse_dates=True)
    else:
        df = yf.download(coin_name, period="1y")
        df.to_csv(f"coins/{coin_name}.csv")
    transpose_df = pd.DataFrame(df[field_name]).transpose()
    transpose_df.insert(loc=0, column='coin_name', value=coin_name)
    transpose_df.set_index('coin_name', inplace=True, drop=True)
    return transpose_df

def download_all_30_coins_dataset ():
    coin_names = { "BTC-USD","ETH-USD","USDT-USD","BNB-USD","SOL-USD","XRP-USD","STETH-USD","USDC-USD","ADA-USD","AVAX-USD","DOGE-USD","TRX-USD",
                "WTRX-USD","LINK-USD","DOT-USD","MATIC-USD","WBTC-USD","TON11419-USD","UNI7083-USD","ICP-USD","SHIB-USD","DAI-USD","BCH-USD",
                "LTC-USD","IMX10603-USD","FIL-USD","ATOM-USD","LEO-USD","KAS-USD","ETC-USD"}
    frames = []
    coin_df = pd.DataFrame()
    if os.path.exists("coins/coins_30_close.csv"):
        top30_coins_df = pd.read_csv(f"coins/coins_30_close.csv", index_col=0, parse_dates=True)
    else:
        for coin in coin_names:
            coin_df = pd.DataFrame()
            coin_df = get_year_data_by_coin_name(coin, "Close")
            frames.append(coin_df)
        top30_coins_df = pd.concat(frames)
        top30_coins_df.dropna(inplace=True)
        top30_coins_df.to_csv("coins/coins_30_close.csv")
        
    return top30_coins_df
        
def download_coin_5Y_data(coin_name):
    coin_df = pd.DataFrame()
    if os.path.exists(f"coins/{coin_name}_5y.csv"):
        coin_df = pd.read_csv(f"coins/{coin_name}_5y.csv", index_col=0, parse_dates=True)
    else:
        coin_df = yf.download(coin_name, interval="1d", period="5y")
        # Drop records with null values
        coin_df = coin_df.dropna()
        coin_df.to_csv(f"coins/{coin_name}_5y.csv")
    # Transform the index to datetime and extract the date
    coin_df.index = pd.to_datetime(coin_df.index).date

    return coin_df

def plot_top_30_corr():
    top30_coins_df = download_all_30_coins_dataset()
    fig = plt.figure(figsize=(12,12))
    sns.heatmap(top30_coins_df.transpose().corr())
    return fig

def create_coins_and_model_dir():
    # Check if coins directory does not exist
    if not os.path.exists("coins/"):
        # Create the coins directory
        os.makedirs("coins/")
    # Check if model directory does not exist
    if not os.path.exists("model/"):
        # Create the model directory
        os.makedirs("model/")
