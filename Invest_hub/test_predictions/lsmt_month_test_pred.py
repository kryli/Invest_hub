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
from keras.src.layers import LSTM, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, GRU
from keras.src.optimizers import Adam, RMSprop
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
import math
from sklearn.metrics import mean_squared_error
import datetime
import mplfinance as mpf
import math 
import os
import plotly.graph_objects as go
import tensorflow as tf

def predict_next_month_stock_price(directory):
    def str_to_datetime(s):
        date_part, time_part = s.split(' ')[0], s.split(' ')[1].split('+')[0]
        year, month, day = map(int, date_part.split('-'))
        hour, minute, second = map(int, time_part.split(':'))
        return datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

    def specific_data(df, start, end):
        return df[(df['Time'] >= start) & (df['Time'] <= end)]

    def process_file(file_path):
        df = pd.read_csv(file_path)
        df['Time'] = df['Time'].apply(str_to_datetime)

        start_time = df['Time'].iloc[0]
        end_time = df['Time'].iloc[-22]
        spec_df = specific_data(df, start_time, end_time)
        new_df = spec_df['Close'].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(new_df.reshape(-1, 1))

        train_size = int(len(scaled_data) * 0.8)
        train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

        n_past = 60
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

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Создание модели
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        checkpoints = ModelCheckpoint(filepath='best_model.keras', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1, callbacks=[checkpoints, early_stopping])

        last_sequence = X_test[-1]
        last_sequence = last_sequence.reshape(1, n_past, 1)

        predictions_next_30_days = []
        for _ in range(30):
            next_day_prediction = model.predict(last_sequence)
            predictions_next_30_days.append(next_day_prediction[0, 0])
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_day_prediction
        predictions_next_30_days = scaler.inverse_transform(np.array(predictions_next_30_days).reshape(-1, 1))

        real_data_next_30_days = specific_data(df, df['Time'].iloc[-22], df['Time'].iloc[-1])
        real_dates = pd.to_datetime(real_data_next_30_days['Time'])
        real_prices = real_data_next_30_days['Close'].values

        # Визуализация
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=real_dates, y=real_prices, mode='lines', name='Actual Prices', line=dict(color='black')))
        fig.add_trace(go.Scatter(x=real_dates[:len(predictions_next_30_days)], y=predictions_next_30_days.flatten(), mode='lines+markers', name='Predicted Prices', line=dict(color='blue')))
        fig.update_layout(
            title=f'Actual vs Predicted Stock Prices for {os.path.basename(file_path)}',
            xaxis_title='Time',
            yaxis_title='Price',
            legend=dict(x=0, y=1),
            template='plotly_white',
            width=800,  # Уменьшаем ширину графика
            height=400,  # Уменьшаем высоту графика
            font=dict(size=10)  # Уменьшаем размер шрифта
        )
        fig.show()

        return predictions_next_30_days

    for filename in os.listdir(directory):
        if "_1d.csv" in filename:
            file_path = os.path.join(directory, filename)
            print(process_file(file_path))

# Пример вызова функции
predict_next_month_stock_price('data/historical_data')