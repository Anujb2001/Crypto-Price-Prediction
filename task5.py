# Task 5.	ML Models for Prediction and Forecasting
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import plotly.express as px
from prophet import Prophet
import pandas as pd
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import sklearn.metrics as metrics
import pickle

f_columns = ['Close']

# create dataset feature and output 
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values # 1,2,3,4.....N       N+1
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def save_model_into_pkl(coin_name, model_name, model):
    with open(f'model/{coin_name}_{model_name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

def load_model_if_exist(coin_name, model_name):
    if os.path.exists(f'model/{coin_name}_{model_name}_model.pkl'):
        with open(f'model/{coin_name}_{model_name}_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model
    else:
        return None
       
def predict_lstm_model(X_test, coin_name):
    #Load the best model
    # The model (that are considered the best) can be loaded as -
    model = keras.models.load_model(f'model/{coin_name}_lstm_model.keras')
    y_pred = model.predict(X_test)
    return y_pred

def split_data_train_test(df, train_percent = 0.95):
      # splitting the dataset
    train_size = int(len(df) * train_percent)
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    return [train, test]

def create_train_test_time_series_data(train_data, test_data, time_steps = 100):
    test_extra = pd.concat([train_data[-time_steps:], test_data], axis=0)
    X_train, y_train = create_dataset(train_data[f_columns], train_data['Close'], time_steps)
    X_test, y_test = create_dataset(test_extra[f_columns], test_extra['Close'], time_steps)
    return [X_train, y_train, X_test, y_test]

def train_and_predict_lstm_model (train, test, coin_name):
    # apply standarization
    f_transformer = StandardScaler()
    c_transformer = StandardScaler()

    f_transformer = f_transformer.fit(train[["Open", "High", "Low"]].to_numpy())
    c_transformer = c_transformer.fit(train[['Close']])
    lstm_train = train.copy()
    lstm_test = test.copy()

    lstm_train.loc[:, ["Open", "High", "Low"]] = f_transformer.transform(lstm_train[["Open", "High", "Low"]].to_numpy())
    lstm_train['Close'] = c_transformer.transform(lstm_train[['Close']])

    lstm_test['Close'] = c_transformer.transform(lstm_test[['Close']])
    
    # Preprocessing
    X_train, y_train, X_test, y_test = create_train_test_time_series_data(lstm_train, lstm_test, 10)
    
    if not os.path.exists(f'model/{coin_name}_lstm_model.keras'):
        # Build the model
        Input_shape = (X_train.shape[1], X_train.shape[2])
        model = keras.Sequential()
        model.add(keras.layers.Bidirectional(
            keras.layers.LSTM(
            units=128,
            input_shape=Input_shape
            )
        )
        )
        model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.Dense(units=1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        #Early stopping and save best models callback
        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')
        Best_Model = keras.callbacks.ModelCheckpoint(f'model/{coin_name}_lstm_model.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

        history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        shuffle=False,
        callbacks=[earlyStopping, Best_Model]
        )
    
    y_pred = predict_lstm_model(X_test, coin_name)
    y_pred_inv = c_transformer.inverse_transform(y_pred.reshape(1, -1)).flatten()
    
    return y_pred_inv

def train_and_predict_Prophet(train_data, test_data, coin_name):
    prophet_model = load_model_if_exist(coin_name,'Prophet')
    if prophet_model == None:
        prophet_df = pd.DataFrame({'ds': train_data.index, 'y': train_data['Close'].values})
        prophet_model = Prophet()
        prophet_model.fit(prophet_df)
        save_model_into_pkl(coin_name, 'Prophet', prophet_model)
    
    y_test = prophet_model.make_future_dataframe(periods=len(test_data))
    prophet_forecast = prophet_model.predict(y_test)
    y_pred = prophet_forecast[-len(test_data):]['yhat']
    return y_pred

def train_and_predict_ARIMA(train_data, test_data, coin_name):
    model = load_model_if_exist(coin_name,'ARIMA')
    if model == None:
        # Fit auto_arima function dataset
        model = auto_arima(train_data['Close'], start_p = 1, start_q = 1,
        max_p = 3, max_q = 3, m = 12,
        start_P = 0, seasonal = True,
        d = None, D = 1, trace = True,
        error_action ='ignore', # we don't want to know if an order does not work
        suppress_warnings = True, # we don't want convergence warnings
        stepwise = True)
        save_model_into_pkl(coin_name,'ARIMA', model) 
    
    y_pred = model.predict(n_periods=test_data.shape[0])
    return y_pred


def train_and_predict_RandomForestRegressor(train_data, test_data, coin_name):
    # Preprocessing
    X_train, y_train, X_test, y_test = create_train_test_time_series_data(train_data, test_data, 10)
    # Random Forest Regressor
    rf_model = load_model_if_exist(coin_name,'Random_Forest')
    if rf_model == None:
        rf_model = RandomForestRegressor(n_estimators=100, random_state=56)  # Example parameters
        rf_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)  # Train the model
        save_model_into_pkl(coin_name,'Random_Forest', rf_model)
        
    y_pred = rf_model.predict(X_test.reshape(X_test.shape[0], -1))    
    return y_pred

def train_and_predict_XGBoost(train_data, test_data, coin_name):
    # Prepare the data
    X_train, y_train, X_test, y_test = create_train_test_time_series_data(train_data, test_data, 10)

    xgb_model = load_model_if_exist(coin_name,'XGBoost')
    if xgb_model == None:
        # Define and train the XGBoost model
        xgb_model = xgb.XGBRegressor()
        xgb_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        save_model_into_pkl(coin_name,'XGBoost', xgb_model)
        
    # Make predictions
    y_pred = xgb_model.predict(X_test.reshape(X_test.shape[0], -1))
    
    return y_pred

def get_evaluation_metrics(y_test, y_pred, model_name):

    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, y_pred)
    return {'Model Name':model_name,
            "Mean Absolute Error (MAE)":mae,
            "Mean Square Error":mse,
            "Root Mean Square Error":rmse,
            "R2 Score":r2}

def plot_prediction(df, coin_name):
    train, test = split_data_train_test(df)
    # train,test_date, y_test, lstm_pred, prophet_pred
    Date_df = pd.DataFrame({'Date':test.index})
    Actual_df = pd.DataFrame({'Actual':test['Close'].values})
    train_data_df = pd.DataFrame({'Date':train.index,'Actual':train['Close']})
    prophet_pred = train_and_predict_Prophet(train, test, coin_name)
    arima_pred = train_and_predict_ARIMA(train, test, coin_name)
    rf_pred = train_and_predict_RandomForestRegressor(train, test, coin_name)
    xg_boost_pred = train_and_predict_XGBoost(train, test, coin_name)
    lstm_pred = train_and_predict_lstm_model(train, test, coin_name)
    
    LSTM_df = pd.DataFrame({'LSTM':lstm_pred})
    prophet_pred_df = pd.DataFrame({'Prophet':prophet_pred.values})
    arima_pred_df = pd.DataFrame({'ARIMA':arima_pred.values})
    rf_pred_df = pd.DataFrame({'Random Forest':rf_pred})
    xg_boost_pred_df = pd.DataFrame({'XGBoost':xg_boost_pred})
    
    
    data = pd.concat([Date_df, Actual_df, LSTM_df, prophet_pred_df, arima_pred_df, rf_pred_df, xg_boost_pred_df ],
                  axis = 1)
    model_pred_columns = ['Actual', 'LSTM', 'Prophet', 'ARIMA', 'Random Forest', 'XGBoost']
    
    evaluation_metrics_list = []
    for model_name in model_pred_columns:
        if model_name != 'Actual':
            evaluation_metrics_list.append(get_evaluation_metrics(data['Actual'], data[model_name], model_name))        
    data = pd.concat([train_data_df, data], axis=0)
    
    fig = px.line(data_frame=data, x='Date', y= model_pred_columns,
                title=f'Prediction of {coin_name} coin using LSTM, Prophet, ARIMA, XGBoost, Random Forest')
    fig.update_yaxes(title_text='Price in USD')
    
    return [fig, pd.DataFrame(evaluation_metrics_list)]
    
    # df = pd.DataFrame({'Column1': array1, 'Column2': array2, 'Column3': array3})

    
    
    






    
    

    
    
    
    