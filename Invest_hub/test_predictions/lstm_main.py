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
import concurrent.futures

def str_to_datetime(s):
    date_part, time_part = s.split(' ')[0], s.split(' ')[1].split('+')[0]
    year, month, day = map(int, date_part.split('-'))
    hour, _, _ = map(int, time_part.split(':'))
    return datetime.datetime(year=year, month=month, day=day, hour=hour, minute=0, second=0)

def specific_data(df, start, end):
    filtered_data = df[(df['Time'] >= start) & (df['Time'] <= end)]
    return filtered_data

def predict_next_day_stock_prices(file_path):
    print(f"Processing file for next day prediction: {file_path}")
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

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

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

    real_data_next_24_hours = specific_data(df, df['Time'].iloc[-15], df['Time'].iloc[-1])
    real_dates = pd.to_datetime(real_data_next_24_hours['Time'])
    stock_name = os.path.basename(file_path).split('_')[0]

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

def predict_next_week_stock_prices(file_path):
    print(f"Processing file for next week prediction: {file_path}")
    df = pd.read_csv(file_path)
    df['Time'] = df['Time'].apply(str_to_datetime)

    start_time = df['Time'].iloc[0]
    end_time = df['Time'].iloc[-35]
    spec_df = specific_data(df, start_time, end_time)
    new_df = spec_df['Close'].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(np.array(new_df).reshape(-1, 1))

    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    n_past = 150
    n_future = 28
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

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(LSTM(units=128))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=n_future))

    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error')

    checkpoints = ModelCheckpoint(filepath='my_weights.keras', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=64,
        verbose=1,
        callbacks=[checkpoints, early_stopping]
    )

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

    real_data_next_7_days = specific_data(df, df['Time'].iloc[-35], df['Time'].iloc[-1])
    real_dates = pd.to_datetime(real_data_next_7_days['Time'])
    real_prices = real_data_next_7_days['Close'].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=real_dates, y=real_prices, mode='lines', name='Actual Prices', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=real_dates[:len(predictions_next_7_days)], y=predictions_next_7_days.flatten(), mode='lines+markers', name='Predicted Prices', line=dict(color='blue')))
    fig.update_layout(
        title=f'Actual vs Predicted Stock Prices for {os.path.basename(file_path)}',
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

    print(predictions_next_7_days)

def predict_next_month_stock_price(file_path):
    print(f"Processing file for next month prediction: {file_path}")
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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=real_dates, y=real_prices, mode='lines', name='Actual Prices', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=real_dates[:len(predictions_next_30_days)], y=predictions_next_30_days.flatten(), mode='lines+markers', name='Predicted Prices', line=dict(color='blue')))
    fig.update_layout(
        title=f'Actual vs Predicted Stock Prices for {os.path.basename(file_path)}',
        xaxis_title='Time',
        yaxis_title='Price',
        legend=dict(x=0, y=1),
        template='plotly_white',
        width=800,
        height=400,
        font=dict(size=10)
    )
    fig.show()

    return predictions_next_30_days

def process_file(file_path):
    try:
        if "_1h.csv" in file_path:
            predict_next_day_stock_prices(file_path)
        elif "_4h.csv" in file_path:
            predict_next_week_stock_prices(file_path)
        elif "_1d.csv" in file_path:
            predict_next_month_stock_price(file_path)
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

def process_directory(directory):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and filename.endswith(".csv"):
                futures.append(executor.submit(process_file, file_path))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

process_directory('test_predictions/historyy')

# import pandas as pd
# import numpy as np

# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import seaborn as sns
# import plotly.express as px
# import warnings
# from sklearn.preprocessing import MinMaxScaler
# from keras import Sequential
# from keras.src.layers import Dense
# from keras.src.layers import LSTM, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, GRU
# from keras.src.optimizers import Adam, RMSprop
# from keras.src.callbacks import ModelCheckpoint, EarlyStopping
# import math
# from sklearn.metrics import mean_squared_error
# import datetime
# import mplfinance as mpf
# import math 
# import os
# import plotly.graph_objects as go
# import tensorflow as tf
# import concurrent.futures
# from keras import Input, Model

# def str_to_datetime(s):
#     date_part, time_part = s.split(' ')[0], s.split(' ')[1].split('+')[0]
#     year, month, day = map(int, date_part.split('-'))
#     hour, _, _ = map(int, time_part.split(':'))
#     return datetime.datetime(year=year, month=month, day=day, hour=hour, minute=0, second=0)

# def specific_data(df, start, end):
#     filtered_data = df[(df['Time'] >= start) & (df['Time'] <= end)]
#     return filtered_data

# def predict_next_day_stock_prices(file_path):
#     df = pd.read_csv(file_path)
#     df['Time'] = df['Time'].apply(str_to_datetime)

#     start_time = df['Time'].iloc[-1350]
#     end_time = df['Time'].iloc[-15]
#     spec_df = specific_data(df, start_time, end_time)
#     new_df = spec_df['Close'].values
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(new_df.reshape(-1, 1))

#     train_size = int(len(scaled_data) * 0.8)
#     train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
#     n_past = 100

#     X_train, y_train = [], []
#     for i in range(n_past, len(train_data)):
#         X_train.append(train_data[i - n_past:i, 0])
#         y_train.append(train_data[i, 0])
#     X_train, y_train = np.array(X_train), np.array(y_train)

#     X_test, y_test = [], []
#     for i in range(n_past, len(test_data)):
#         X_test.append(test_data[i - n_past:i, 0])
#         y_test.append(test_data[i, 0])
#     X_test, y_test = np.array(X_test), np.array(y_test)

#     X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#     X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#     model = Sequential()
#     model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=128, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=128))
#     model.add(Dropout(0.2))
#     model.add(Dense(units=1))
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
#     checkpoints = ModelCheckpoint(filepath='my_weights.keras', save_best_only=True)
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=0, callbacks=[checkpoints, early_stopping])

#     last_sequence = X_test[-1].reshape(1, n_past, 1)
#     predictions_next_24_hours = []
#     for _ in range(24):
#         next_hour_prediction = model.predict(last_sequence)
#         predictions_next_24_hours.append(next_hour_prediction[0, 0])
#         last_sequence = np.roll(last_sequence, -1, axis=1)
#         last_sequence[0, -1, 0] = next_hour_prediction

#     predictions_next_24_hours = scaler.inverse_transform(np.array(predictions_next_24_hours).reshape(-1, 1))
#     return predictions_next_24_hours[-1][0]

# def predict_next_week_stock_prices(file_path):
#     df = pd.read_csv(file_path)
#     df['Time'] = df['Time'].apply(str_to_datetime)

#     start_time = df['Time'].iloc[0]
#     end_time = df['Time'].iloc[-35]
#     spec_df = specific_data(df, start_time, end_time)
#     new_df = spec_df['Close'].values
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(np.array(new_df).reshape(-1, 1))

#     train_size = int(len(scaled_data) * 0.8)
#     train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

#     n_past = 150  # Увеличено окно
#     n_future = 28  # Прогнозируем 7 дней (28 * 4 часа)
#     X_train, y_train = [], []
#     for i in range(n_past, len(train_data) - n_future):
#         X_train.append(train_data[i - n_past:i, 0])
#         y_train.append(train_data[i:i + n_future, 0])
#     X_train, y_train = np.array(X_train), np.array(y_train)

#     X_test, y_test = [], []
#     for i in range(n_past, len(test_data) - n_future):
#         X_test.append(test_data[i - n_past:i, 0])
#         y_test.append(test_data[i:i + n_future, 0])
#     X_test, y_test = np.array(X_test), np.array(y_test)

#     X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#     X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#     model = Sequential()
#     model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Dropout(0.3))
#     model.add(BatchNormalization())

#     model.add(LSTM(units=128, return_sequences=True))
#     model.add(Dropout(0.3))
#     model.add(BatchNormalization())
#     model.add(LSTM(units=128))
#     model.add(Dropout(0.3))
#     model.add(BatchNormalization())

#     model.add(Dense(units=64, activation='relu'))
#     model.add(Dense(units=n_future))

#     model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error')

#     checkpoints = ModelCheckpoint(filepath='my_weights.keras', save_best_only=True)
#     early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
#     model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=0, callbacks=[checkpoints, early_stopping])

#     last_sequence = X_test[-1]
#     look_back = 150
#     last_sequence = last_sequence.reshape(1, look_back, 1)

#     predictions_next_7_days = []
#     for _ in range(7):  # 7 дней
#         daily_predictions = []
#         for _ in range(4):  # 4 часа в день
#             next_hour_prediction = model.predict(last_sequence)
#             daily_predictions.append(next_hour_prediction[0, 0])
#             last_sequence = np.roll(last_sequence, -1, axis=1)
#             last_sequence[0, -1, 0] = next_hour_prediction[0, 0]  # Преобразование в скалярное значение
#         predictions_next_7_days.extend(daily_predictions)

#     predictions_next_7_days = scaler.inverse_transform(np.array(predictions_next_7_days).reshape(-1, 1))
#     return predictions_next_7_days[-1][0]

# def predict_next_month_stock_price(file_path):
#     df = pd.read_csv(file_path)
#     df['Time'] = df['Time'].apply(str_to_datetime)

#     start_time = df['Time'].iloc[0]
#     end_time = df['Time'].iloc[-22]
#     spec_df = specific_data(df, start_time, end_time)
#     new_df = spec_df['Close'].values
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(new_df.reshape(-1, 1))

#     train_size = int(len(scaled_data) * 0.8)
#     train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

#     n_past = 60
#     X_train, y_train = [], []
#     for i in range(n_past, len(train_data)):
#         X_train.append(train_data[i - n_past:i, 0])
#         y_train.append(train_data[i, 0])
#     X_train, y_train = np.array(X_train), np.array(y_train)

#     X_test, y_test = [], []
#     for i in range(n_past, len(test_data)):
#         X_test.append(test_data[i - n_past:i, 0])
#         y_test.append(test_data[i, 0])
#     X_test, y_test = np.array(X_test), np.array(y_test)

#     X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#     X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=50, return_sequences=False))
#     model.add(Dropout(0.2))
#     model.add(Dense(units=1))

#     model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
#     checkpoints = ModelCheckpoint(filepath='best_model.keras', save_best_only=True)
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=0, callbacks=[checkpoints, early_stopping])

#     last_sequence = X_test[-1]
#     last_sequence = last_sequence.reshape(1, n_past, 1)

#     predictions_next_30_days = []
#     for _ in range(30):
#         next_day_prediction = model.predict(last_sequence)
#         predictions_next_30_days.append(next_day_prediction[0, 0])
#         last_sequence = np.roll(last_sequence, -1, axis=1)
#         last_sequence[0, -1, 0] = next_day_prediction
#     predictions_next_30_days = scaler.inverse_transform(np.array(predictions_next_30_days).reshape(-1, 1))
#     return predictions_next_30_days[-1][0]

# def find_files_for_ticker(ticker, directory='data/historical_data'):
#     file_path_1h = None
#     file_path_4h = None
#     file_path_1d = None
    
#     for filename in os.listdir(directory):
#         if filename.startswith(ticker):
#             if filename.endswith("_1h.csv"):
#                 file_path_1h = os.path.join(directory, filename)
#             elif filename.endswith("_4h.csv"):
#                 file_path_4h = os.path.join(directory, filename)
#             elif filename.endswith("_1d.csv"):
#                 file_path_1d = os.path.join(directory, filename)
    
#     return file_path_1h, file_path_4h, file_path_1d

# def predict_stock_prices(ticker):
#     file_path_1h, file_path_4h, file_path_1d = find_files_for_ticker(ticker)
    
#     if not file_path_1h or not file_path_4h or not file_path_1d:
#         raise FileNotFoundError("One or more necessary files are missing for the ticker: " + ticker)
    
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future_day = executor.submit(predict_next_day_stock_prices, file_path_1h)
#         future_week = executor.submit(predict_next_week_stock_prices, file_path_4h)
#         future_month = executor.submit(predict_next_month_stock_price, file_path_1d)

#         day_price = future_day.result()
#         week_price = future_week.result()
#         month_price = future_month.result()
    
#     return [day_price, week_price, month_price]

# # Пример использования
# ticker = 'FLOT'  # Укажите тикер акции
# predicted_prices = predict_stock_prices(ticker)
# print(predicted_prices)

###
###2 variant
###