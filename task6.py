# Task 6.	General Trading Signals and AnalysisForecasting
import task5 as models
import pandas as pd
from datetime import timedelta
import plotly.express as px


def get_forecast_value(train, forecasting_test_data, coin_name, forecasting_model):
    if forecasting_model == 'Random Forest':
        y_pred = models.train_and_predict_RandomForestRegressor(train, forecasting_test_data, coin_name)
    elif forecasting_model == 'LSTM':
        y_pred = models.train_and_predict_lstm_model(train, forecasting_test_data, coin_name)
    elif forecasting_model == 'ARIMA':
        y_pred = models.train_and_predict_ARIMA(train, forecasting_test_data, coin_name).values
    elif forecasting_model == 'Prophet':
        y_pred = models.train_and_predict_Prophet(train, forecasting_test_data, coin_name).values
    elif forecasting_model == 'XGBoost':
        y_pred = models.train_and_predict_XGBoost(train, forecasting_test_data, coin_name)
    return y_pred
            
def forecast(coin_name, coin_df, forecasting_model, forecast_period):
    train = coin_df.copy()
    latest_date = coin_df.index[-1]
    forecasting_dates = []
    latest_price = coin_df['Close'][-1]
    y_pred =[]
    for day in range(forecast_period):
        forecasting_dates.append(latest_date + timedelta(days=day))
    forecasting_test_data = pd.DataFrame(data={'Close': latest_price}, index=forecasting_dates)
    y_pred = get_forecast_value(train, forecasting_test_data, coin_name, forecasting_model)
    
    fig = plot_forecasting(coin_df[-60:], y_pred, forecasting_dates, forecasting_model, coin_name)
    
    return [y_pred, fig, latest_price]
    
def plot_forecasting(df, forecast, forecasting_dates, model_name, coin_name):
    model_df = pd.DataFrame(data = {'Forecasting':forecast, 'Actual':"",'Date':forecasting_dates}, index=forecasting_dates)
    Actual_df = pd.DataFrame(data={'Forecasting': "", 'Actual':df['Close'].values, 'Date':df.index}, index=df.index)
    model_pred_columns = ['Actual', 'Forecasting']
    data = pd.concat([Actual_df, model_df], axis=0)
    fig = px.line(data_frame=data, x='Date', y= model_pred_columns,
                title=f'Forecasting of {coin_name} coin using {model_name}')
    fig.update_yaxes(title_text='Price in USD')
    
    return fig
     
    