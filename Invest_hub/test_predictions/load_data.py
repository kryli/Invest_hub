import os
from datetime import datetime, timedelta
from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.exceptions import RequestError
import csv
import schedule
import time

TOKEN = 't.wubN8WNCl0M-LZ4aziTOZ8PZ2EXc9RIUWjpy2_C3uUEntYYYWv7NAkMIoSasAtW-dHc-f3JDdZ8vnKAYosNV9Q'
output_folder = 'test_predictions/historyy'

def get_figi_by_ticker(ticker):
    with Client(TOKEN) as client:
        for method in ['shares', 'etfs', 'bonds', 'currencies', 'futures']:
            for item in getattr(client.instruments, method)().instruments:
                if item.ticker == ticker:
                    return item.figi
        print(f"No FIGI found for ticker {ticker}")
        return None

def fetch_candles(figi, start_date, end_date, interval):
    with Client(TOKEN) as client:
        try:
            candles = client.get_all_candles(
                figi=figi,
                from_=start_date,
                to=end_date,
                interval=interval,
            )
            return list(candles)
        except RequestError as e:
            print(f"An error occurred: {e}")
            return []

def write_candles_to_csv(candles, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        for candle in candles:
            writer.writerow([
                candle.time,
                candle.open.units + candle.open.nano / 1e9,
                candle.high.units + candle.high.nano / 1e9,
                candle.low.units + candle.low.nano / 1e9,
                candle.close.units + candle.close.nano / 1e9,
                candle.volume,
            ])

def process_single_ticker(ticker):
    figi = get_figi_by_ticker(ticker)
    if figi:
        print(f"Processing {ticker} with FIGI {figi}")
        
        start_date_1d = datetime.now() - timedelta(days=1095)
        candles = fetch_candles(figi, start_date_1d, datetime.now(), CandleInterval.CANDLE_INTERVAL_DAY)
        write_candles_to_csv(candles, os.path.join(output_folder, f'{ticker}_1d.csv'))
        
        start_date_4h = datetime.now() - timedelta(days=730)
        candles = fetch_candles(figi, start_date_4h, datetime.now(), CandleInterval.CANDLE_INTERVAL_4_HOUR)
        write_candles_to_csv(candles, os.path.join(output_folder, f'{ticker}_4h.csv'))
        
        start_date_1h = datetime.now() - timedelta(days=365)
        candles = fetch_candles(figi, start_date_1h, datetime.now(), CandleInterval.CANDLE_INTERVAL_HOUR)
        write_candles_to_csv(candles, os.path.join(output_folder, f'{ticker}_1h.csv'))
    else:
        print(f"Failed to process {ticker}")

def process_tickers_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    ticker = line.strip().split(',')[0]
                    process_single_ticker(ticker)

def daily_update():
    print("Starting daily update...")
    process_tickers_in_folder(data_folder)
    print("Daily update completed.")

def hourly_update():
    print("Starting hourly update...")
    process_tickers_in_folder(data_folder)
    print("Hourly update completed.")

def every_4_hour_update():
    print("Starting every 4 hour update...")
    process_tickers_in_folder(data_folder)
    print("Every 4 hour update completed.")

data_folder = 'test_predictions/tikkers'

process_tickers_in_folder(data_folder)

schedule.every().day.at("03:00").do(daily_update)

schedule.every().hour.do(hourly_update)

schedule.every(4).hours.do(every_4_hour_update)

while True:
    schedule.run_pending()
    time.sleep(1)
