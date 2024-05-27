import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import warnings
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.src.layers import Dense
from keras.src.layers import LSTM, Dropout
from keras.src.optimizers import Adam
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
import math
from sklearn.metrics import mean_squared_error
import datetime
import mplfinance as mpf
import math 
import os
import plotly.graph_objects as go

def str_to_datetime(s):
    date_part, time_part = s.split(' ')[0], s.split(' ')[1].split('+')[0]
    year, month, day = map(int, date_part.split('-'))
    hour, _, _ = map(int, time_part.split(':'))
    return datetime.datetime(year=year, month=month, day=day, hour=hour, minute=0, second=0)

def specific_data(df, start, end):
    filtered_data = df[(df['Time'] >= start) & (df['Time'] <= end)]
    return filtered_data

def predict_next_day_stock_prices(directory):
    for filename in os.listdir(directory):
        if "_1h.csv" in filename:
            file_path = os.path.join(directory, filename)
            
            # Загрузка данных
            df = pd.read_csv(file_path)
            df['Time'] = df['Time'].apply(str_to_datetime)

            start_time = df['Time'].iloc[-1350]
            end_time = df['Time'].iloc[-15]
            spec_df = specific_data(df, start_time, end_time)
            new_df = spec_df['Close'].values
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(new_df.reshape(-1, 1))

            train_size = int(len(scaled_data) * 0.8)
            train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
            n_past = 100

            X_train, y_train = [], []
            for i in range(n_past, len(train_data)):
                X_train.append(train_data[i - n_past:i, 0])
                y_train.append(train_data[i, 0])
            X_train, y_train = np.array(X_train), np.array(y_train)

            X_test, y_test = [], []
            for i in range(n_past, len(test_data)):
                X_test.append(test_data[i - n_past:i, 0])
                y_test.append(test_data[i, 0])
            X_test, y_test = np.array(X_test), np.array(y_test)

            # Изменение формы данных для LSTM
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # Построение и обучение модели LSTM
            model = Sequential()
            model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=128, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=128))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            checkpoints = ModelCheckpoint(filepath='my_weights.keras', save_best_only=True)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1, callbacks=[checkpoints, early_stopping])

            last_sequence = X_test[-1].reshape(1, n_past, 1)
            predictions_next_24_hours = []
            for _ in range(24):
                next_hour_prediction = model.predict(last_sequence)
                predictions_next_24_hours.append(next_hour_prediction[0, 0])
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = next_hour_prediction

            predictions_next_24_hours = scaler.inverse_transform(np.array(predictions_next_24_hours).reshape(-1, 1))

            # График с использованием Plotly
            real_data_next_24_hours = specific_data(df, df['Time'].iloc[-15], df['Time'].iloc[-1])
            real_dates = pd.to_datetime(real_data_next_24_hours['Time'])
            stock_name = filename.split('_')[0]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=real_dates, y=real_data_next_24_hours['Close'], mode='lines', name=f'Actual Prices ({stock_name})', line=dict(color='black')))
            fig.add_trace(go.Scatter(x=real_dates[:len(predictions_next_24_hours)], y=predictions_next_24_hours.flatten(), mode='lines+markers', name=f'Predicted Prices ({stock_name})', line=dict(color='blue')))
            fig.update_layout(
                title=f'Actual vs Predicted Stock Prices for {stock_name}',
                xaxis_title='Time',
                yaxis_title='Price',
                legend=dict(x=0, y=1),
                template='plotly_white',
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1d", step="day", stepmode="backward"),
                            dict(count=7, label="7d", step="day", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                ),
                yaxis=dict(
                    autorange=True,
                    fixedrange=False
                )
            )
            fig.show()

predict_next_day_stock_prices('data/historical_data')