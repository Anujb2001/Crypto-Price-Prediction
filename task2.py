# Task 2.	Grouping: Apply Clustering to group the 30 cryptocurrencies into 4 groups
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def reduce_dimension(df):
    scaler = Normalizer()
    scaled_df = scaler.fit_transform(df)
    pca_obj = PCA(n_components=10)
    pca_df = pd.DataFrame(pca_obj.fit_transform(scaled_df))
    pca_df["coin_name"] = df.index
    pca_df.set_index("coin_name",drop=True, inplace=True)
    return pca_df

def group_coins(df):
    kmeans = KMeans(n_clusters = 4, random_state = 80)
    kmeans.fit(df)
    y_kmeans=kmeans.fit_predict(df)
    new_df = df.copy()
    new_df.insert(0, "group_index", kmeans.labels_, True)
    return new_df

def display_group_coin_data (df):
    pca_df = reduce_dimension(df)

    group_df= group_coins(pca_df)
    group_df = group_df.sort_values('group_index')
    # Assign colors based on 'group_index'
    color_mapping = {0: 'blue', 1: 'green', 2: 'red', 3: 'purple'}
    group_df['color'] = group_df['group_index'].map(color_mapping)
    # Create a list of patches for the legend
    patches = [mpatches.Patch(color=color, label=f'Group - {index}') for index, color in color_mapping.items()]
    # Plot the bar chart
    fig = plt.figure(figsize=(10, 3))
    plt.bar(group_df.index, 1, color=group_df['color'])
    plt.xlabel('Coin Name')
    plt.xticks(rotation=90)
    plt.yticks([])
    plt.title('4 Group of Cryptocurrencies')
    plt.legend(handles=patches, title='Color Legend')
    return fig

def get_four_coin_each_group():
    return {"group1":"SOL-USD", 
            "group2":"BNB-USD", 
            "group3":"LTC-USD", 
            "group4":"BCH-USD"}
    

    
    