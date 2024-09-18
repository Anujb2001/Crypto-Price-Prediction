# Task 7.	GUI- Develop UI by using streamlit
import streamlit as st
from streamlit_option_menu import option_menu
import task1 as data_loader
import task2 as clust_df
import task3 as correlation
import task4 as chart
import task5 as models
import task6 as forecasting

def load_side_menu():
    with st.sidebar:
        selected = option_menu(
        menu_title="Main Menu",  # required
        options=["Top 30 Coins",
                 "PCA and Grouping", 
                 "Correlation","EDA", 
                 "Prediction", 
                 "Forecasting"],  # required
        # icons=["house", "book", "envelope"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
    )
    return selected
 
def display_download_page(selected_option):
    if selected_option == "Top 30 Coins":
        st.title(selected_option)
        fig = data_loader.plot_top_30_corr()
        st.pyplot(fig)


def display_grouping_page(selected_option):
    if selected_option == "PCA and Grouping":
        fig = clust_df.display_group_coin_data(data_loader.download_all_30_coins_dataset())
        st.pyplot(fig)

def display_correlation_page(selected_option):
    coins = clust_df.get_four_coin_each_group()
    if selected_option == "Correlation":
        coin_name = st.selectbox('Select Coin', coins.values())
        df = data_loader.download_all_30_coins_dataset()
        top_4_selection = st.selectbox('Select Top-4 highly', ["Positive correlated", "Negative correlated"] )
        fig = correlation.plot_top4_correlated(df,coin_name, top_4_selection == "Positive correlated")    
        st.pyplot(fig)
    
def display_EDA_page(selected_option):
    if selected_option == "EDA":
        st.title('Exploratory Data Analysis(EDA)')
        eda_chart_options = [
            'Seasonal Decomposition',
            'Cryptocurrency Price Over Time',
            'Distribution of Cryptocurrency Prices',
            'Box Plot Monthly Distribution',
            'Time Series & Autocorrelation',
            'Scatter lag plot',
            'Plot Moving Average'
        ]
        
        chart_name = st.selectbox('Select EDA Chart', eda_chart_options)
        coins = clust_df.get_four_coin_each_group()
        
        if chart_name == "Seasonal Decomposition":
           coin_name = st.selectbox('Select Coin', coins.values())
           coin_df = data_loader.download_coin_5Y_data(coin_name)
           fig = chart.plot_seasonal_decompose(coin_df)
           st.plotly_chart(fig)
        if chart_name == "Cryptocurrency Price Over Time":
            coins_data = {}
            for coin in coins.values():
                coins_data[coin] = data_loader.download_coin_5Y_data(coin)
            fig = chart.plot_line_chart_price_over_time(coins_data)
            st.plotly_chart(fig)
        if chart_name == "Distribution of Cryptocurrency Prices":
            coins_data = {}
            for coin in coins.values():
                coins_data[coin] = data_loader.download_coin_5Y_data(coin)
            fig = chart.plot_distribution_of_coin_prices(coins_data)
            st.plotly_chart(fig)
        if chart_name == "Box Plot Monthly Distribution":
           coin_name = st.selectbox('Select Coin', coins.values())
           coin_df = data_loader.download_coin_5Y_data(coin_name)
           fig = chart.box_plot_monthly_distrb(coin_df, coin_name)
           st.pyplot(fig)
        if chart_name == "Time Series & Autocorrelation":
           coin_name = st.selectbox('Select Coin', coins.values())
           coin_df = data_loader.download_coin_5Y_data(coin_name)
           fig = chart.plot_time_series_autocorrelation(coin_df, coin_name)
           st.pyplot(fig)
        if chart_name == "Scatter lag plot":
           coin_name = st.selectbox('Select Coin', coins.values())
           coin_df = data_loader.download_coin_5Y_data(coin_name)
           fig = chart.scatter_plot_lag(coin_df, coin_name)
           st.pyplot(fig)
        if chart_name == "Plot Moving Average":
           coin_name = st.selectbox('Select Coin', coins.values())
           coin_df = data_loader.download_coin_5Y_data(coin_name)
           fig = chart.plot_moving_average(coin_df, coin_name)
           st.pyplot(fig)
    
def display_prediction_page(selected_option):
    coins = clust_df.get_four_coin_each_group()
    if selected_option == "Prediction":
        st.title(selected_option)
        coin_name = st.selectbox('Select Coin', coins.values())
        coin_df = data_loader.download_coin_5Y_data(coin_name)
        fig, evaluation_df = models.plot_prediction(coin_df, coin_name)
        st.plotly_chart(fig)
        # Create a Streamlit app
        st.title('Model Performance Metrics')

        # Display a table
        st.subheader('Model Evaluation metrics')
        st.table(evaluation_df)

        # Display a chart
        evaluation_metrics = st.selectbox('Select evaluation metrics', 
                                          ["Mean Absolute Error (MAE)", 
                                           "Mean Square Error", 
                                           "Root Mean Square Error", 
                                           "R2 Score"])
        
        st.subheader(evaluation_metrics)
        st.bar_chart(evaluation_df.set_index('Model Name')[evaluation_metrics])

def display_forecasting_page(selected_option):
    coins = clust_df.get_four_coin_each_group()
    if selected_option == "Forecasting":
       st.title(selected_option)
       coin_name = st.selectbox('Select Coin', coins.values())
       coin_df = data_loader.download_coin_5Y_data(coin_name)
       forecasting_model = st.selectbox('Select forecasting model', 
                                          ['LSTM', 'Prophet', 'ARIMA', 'Random Forest', 'XGBoost'])
       forecast_period = int(st.selectbox('Select forecast period in days', ["7", "14", "30"]))
       y_pred, fig, latest_price = forecasting.forecast(coin_name, coin_df, forecasting_model, forecast_period)
    #    st.write(y_pred)
       if latest_price <= y_pred[-1]:
           st.info(f'Good time to "BUY" the {coin_name} coin')
       else:
           st.info(f'Good time to "SELL" the {coin_name} coin')
       st.plotly_chart(fig)
       
       
         
            
          

  