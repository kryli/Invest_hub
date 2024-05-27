import pandas as pd
import numpy as np
import random
import datetime

df = pd.read_csv('data/historical_data/AFKS_1d.csv', names=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'], skiprows=1)

def str_to_datetime(s):
    date_part, time_part = s.split(' ')[0], s.split(' ')[1].split('+')[0]
    year, month, day = map(int, date_part.split('-'))
    hour, _, _ = map(int, time_part.split(':'))
    return datetime.datetime(year=year, month=month, day=day, hour=hour, minute=0, second=0)

df['Time'] = df['Time'].apply(str_to_datetime)

last_30_days_df = df[-30:]

def generate_random_decisions(num_days):
    decisions = []
    for _ in range(num_days):
        decision = random.choice(["buy", "sell"])
        decisions.append(decision)
    return decisions

def calculate_profit(df, decisions):
    capital = 0
    shares = 0
    for i in range(len(decisions)):
        if decisions[i] == "buy":
            shares += 1
            capital -= df.iloc[i]["Close"]
        elif decisions[i] == "sell" and shares > 0:
            shares -= 1
            capital += df.iloc[i]["Close"]
    
    if shares > 0:
        capital += shares * df.iloc[-1]["Close"]
    
    return capital

def average_profit_over_simulations(df, num_simulations, num_days):
    total_profit = 0
    for _ in range(num_simulations):
        random_decisions = generate_random_decisions(num_days)
        profit = calculate_profit(df, random_decisions)
        total_profit += profit
    average_profit = total_profit / num_simulations
    return round(average_profit, 6)

num_simulations = 20
num_days = 30

average_profit = average_profit_over_simulations(last_30_days_df, num_simulations, num_days)
print("Average profit after 30 days over 20 simulations:", average_profit)
