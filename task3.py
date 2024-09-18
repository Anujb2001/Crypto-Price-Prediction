# Task 3.	Correlation Present Top-4 Highly Correlated (Positive and Negative) for the 4 cryptocurrencies selected.
import seaborn as sns
import matplotlib.pyplot as plt

  
def plot_top4_correlated(df, coin_name, positive=True):
    corr_matrix = df.transpose().corr()

    # Now, isolate the 'BTC-USD' correlations
    btc_usd_corr = corr_matrix[coin_name].sort_values(ascending = not positive).head(5)[1:].to_frame()

    # Plot the heatmap
    fig = plt.figure(figsize=(10,3))
    sns.heatmap(btc_usd_corr, annot=True, cmap='summer')
    if positive :
        plt.title(f'{coin_name} Top-4 Highly Positive Correlated ')
    else:
        plt.title(f'{coin_name} Top-4 Highly Negative or Least Correlated ')
    return fig    
