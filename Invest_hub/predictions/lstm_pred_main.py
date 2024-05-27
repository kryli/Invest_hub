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
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, Callback
import math
from sklearn.metrics import mean_squared_error
import datetime
import mplfinance as mpf
import math 
import os
import plotly.graph_objects as go
import tensorflow as tf
import concurrent.futures
from keras import Input, Model
import json
from tqdm import tqdm
import logging
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.getLogger('tensorflow').setLevel(logging.ERROR)

class SuppressTFOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

class QuietCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

def str_to_datetime(s):
    date_part, time_part = s.split(' ')[0], s.split(' ')[1].split('+')[0]
    year, month, day = map(int, date_part.split('-'))
    hour, _, _ = map(int, time_part.split(':'))
    return datetime.datetime(year=year, month=month, day=day, hour=hour, minute=0, second=0)

def specific_data(df, start, end):
    filtered_data = df[(df['Time'] >= start) & (df['Time'] <= end)]
    return filtered_data

def create_day_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(units=128, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(units=128, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(units=128)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(units=1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def predict_next_day_stock_prices(file_path):
    df = pd.read_csv(file_path)
    df['Time'] = df['Time'].apply(str_to_datetime)

    start_time = df['Time'].iloc[-1350]
    end_time = df['Time'].iloc[-1]
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

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = create_day_model((X_train.shape[1], 1))
    checkpoints = ModelCheckpoint(filepath='my_weights.keras', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=0, callbacks=[checkpoints, early_stopping, QuietCallback()])

    last_sequence = X_test[-1].reshape(1, n_past, 1)
    predictions_next_24_hours = []
    for _ in range(24):
        next_hour_prediction = model.predict(last_sequence)
        predictions_next_24_hours.append(next_hour_prediction[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = next_hour_prediction[0, 0]

    predictions_next_24_hours = scaler.inverse_transform(np.array(predictions_next_24_hours).reshape(-1, 1))
    return float(predictions_next_24_hours[-1][0])

def create_week_model(input_shape, n_future):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=2, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    x = LSTM(units=128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = LSTM(units=128)(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    x = Dense(units=64, activation='relu')(x)
    outputs = Dense(units=n_future)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error')
    return model

def predict_next_week_stock_prices(file_path):
    df = pd.read_csv(file_path)
    df['Time'] = df['Time'].apply(str_to_datetime)

    start_time = df['Time'].iloc[0]
    end_time = df['Time'].iloc[-1]
    spec_df = specific_data(df, start_time, end_time)
    new_df = spec_df['Close'].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(np.array(new_df).reshape(-1, 1))

    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    n_past = 150  # Увеличено окно
    n_future = 28  # Прогнозируем 7 дней (28 * 4 часа)
    X_train, y_train = [], []
    for i in range(n_past, len(train_data) - n_future):
        X_train.append(train_data[i - n_past:i, 0])
        y_train.append(train_data[i:i + n_future, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_test, y_test = [], []
    for i in range(n_past, len(test_data) - n_future):
        X_test.append(test_data[i - n_past:i, 0])
        y_test.append(test_data[i:i + n_future, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = create_week_model((X_train.shape[1], 1), n_future)
    checkpoints = ModelCheckpoint(filepath='my_weights.keras', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=0, callbacks=[checkpoints, early_stopping, QuietCallback()])

    last_sequence = X_test[-1]
    look_back = 150
    last_sequence = last_sequence.reshape(1, look_back, 1)

    predictions_next_7_days = []
    for _ in range(7):
        daily_predictions = []
        for _ in range(4):
            next_hour_prediction = model.predict(last_sequence)
            daily_predictions.append(next_hour_prediction[0, 0])
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_hour_prediction[0, 0]
        predictions_next_7_days.extend(daily_predictions)

    predictions_next_7_days = scaler.inverse_transform(np.array(predictions_next_7_days).reshape(-1, 1))
    return float(predictions_next_7_days[-1][0])

def create_month_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(units=50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(units=50, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(units=1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def predict_next_month_stock_price(file_path):
    df = pd.read_csv(file_path)
    df['Time'] = df['Time'].apply(str_to_datetime)

    start_time = df['Time'].iloc[0]
    end_time = df['Time'].iloc[-1]
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

    model = create_month_model((X_train.shape[1], 1))
    checkpoints = ModelCheckpoint(filepath='best_model.keras', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=0, callbacks=[checkpoints, early_stopping, QuietCallback()])

    last_sequence = X_test[-1]
    last_sequence = last_sequence.reshape(1, n_past, 1)

    predictions_next_30_days = []
    for _ in range(30):
        next_day_prediction = model.predict(last_sequence)
        predictions_next_30_days.append(next_day_prediction[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = next_day_prediction[0, 0]
    predictions_next_30_days = scaler.inverse_transform(np.array(predictions_next_30_days).reshape(-1, 1))
    return float(predictions_next_30_days[-1][0])

def find_files_for_ticker(ticker, directory='/Users/leo/Desktop/Invest_hub/data/historical_data'):
    file_path_1h = None
    file_path_4h = None
    file_path_1d = None
    
    for filename in os.listdir(directory):
        if filename.startswith(ticker):
            if filename.endswith("_1h.csv"):
                file_path_1h = os.path.join(directory, filename)
            elif filename.endswith("_4h.csv"):
                file_path_4h = os.path.join(directory, filename)
            elif filename.endswith("_1d.csv"):
                file_path_1d = os.path.join(directory, filename)
    
    return file_path_1h, file_path_4h, file_path_1d

def calculate_percentage_change(start_price, predicted_price):
    percentage_change = ((predicted_price - start_price) / start_price) * 100
    formatted_change = f"{'+' if percentage_change > 0 else ''}{percentage_change:.2f}%"
    return formatted_change


def predict_stock_prices(ticker):
    file_path_1h, file_path_4h, file_path_1d = find_files_for_ticker(ticker)
    
    if not file_path_1h or not file_path_4h or not file_path_1d:
        raise FileNotFoundError("One or more necessary files are missing for the ticker: " + ticker)
    
    df = pd.read_csv(file_path_1h)
    df['Time'] = df['Time'].apply(str_to_datetime)
    start_price = df['Close'].iloc[-1]  # Начальная цена для расчета процента изменения

    with concurrent.futures.ThreadPoolExecutor() as executor:
        with SuppressTFOutput():
            future_day = executor.submit(predict_next_day_stock_prices, file_path_1h)
            future_week = executor.submit(predict_next_week_stock_prices, file_path_4h)
            future_month = executor.submit(predict_next_month_stock_price, file_path_1d)

            day_price = float(future_day.result())
            week_price = float(future_week.result())
            month_price = float(future_month.result())
    
    day_change = calculate_percentage_change(start_price, day_price)
    week_change = calculate_percentage_change(start_price, week_price)
    month_change = calculate_percentage_change(start_price, month_price)

    predictions = {
    "day": {"price": round(day_price, 2), "change": day_change},
    "week": {"price": round(week_price, 2), "change": week_change},
    "month": {"price": round(month_price, 2), "change": month_change}
    }

    print(json.dumps(predictions))

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1]
    predicted_prices = predict_stock_prices(ticker)