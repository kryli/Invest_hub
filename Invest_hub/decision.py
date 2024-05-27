import pandas_ta as ta
import pandas as pd
import numpy as np
import sys
import warnings

def main(ticker):
    warnings.simplefilter('ignore')
    file_path = f'//Users/leo/Desktop/Invest_hub/data/historical_data/{ticker}_1d.csv'
    df = pd.read_csv(file_path)

    CustomStrategy = ta.Strategy(
        name="Custom Technical Indicators",
        description="SMA for short, medium, and long term periods, RSI, STOCH, STOCHRSI, MACD, ADX, Williams %R, CCI, ATR, Ultimate Oscillator, ROC, and Bull/Bear Power",
        ta=[
            {"kind": "sma", "length": 5},  # Short-term SMA
            {"kind": "sma", "length": 10},
            {"kind": "sma", "length": 20},
            {"kind": "sma", "length": 50},  # Medium-term SMA
            {"kind": "sma", "length": 100},
            {"kind": "sma", "length": 200},  # Long-term SMA
            {"kind": "rsi", "length": 14},
            {"kind": "stoch", "k": 9, "d": 6},
            {"kind": "stochrsi", "length": 14},
            {"kind": "macd", "fast": 12, "slow": 26},
            {"kind": "adx", "length": 14},
            {"kind": "willr", "length": 14},
            {"kind": "cci", "length": 14},
            {"kind": "atr", "length": 14},
            {"kind": "uo", "fast": 7, "medium": 14, "slow": 28},  # Ultimate Oscillator
            {"kind": "roc", "length": 14},
            {"kind": "psar"}  # Assuming Bull/Bear Power is represented by Parabolic SAR (psar)
        ]
    )
    df.ta.strategy(CustomStrategy, cores=1)

    def make_decision(df):
        buy_signals = 0
        sell_signals = 0
        neutral_signals = 0
        volatility_signal = "Neutral"

        # RSI
        if df['RSI_14'].iloc[-1] > 70:
            sell_signals += 1
        elif df['RSI_14'].iloc[-1] < 30:
            buy_signals += 1
        else:
            neutral_signals += 1

        # Stochastic Oscillator
        if df['STOCHk_9_6_3'].iloc[-1] > 80:
            sell_signals += 1
        elif df['STOCHk_9_6_3'].iloc[-1] < 20:
            buy_signals += 1
        else:
            neutral_signals += 1

        # Stochastic RSI
        if df['STOCHRSIk_14_14_3_3'].iloc[-1] > 0.8:
            sell_signals += 1
        elif df['STOCHRSIk_14_14_3_3'].iloc[-1] < 0.2:
            buy_signals += 1
        else:
            neutral_signals += 1

        # MACD
        if df['MACDh_12_26_9'].iloc[-1] > 0:  # Histogram positive: buy signal
            buy_signals += 1
        elif df['MACDh_12_26_9'].iloc[-1] < 0:  # Histogram negative: sell signal
            sell_signals += 1
        else:
            neutral_signals += 1

        # ADX
        if df['ADX_14'].iloc[-1] > 25:
            if df['DMP_14'].iloc[-1] > df['DMN_14'].iloc[-1]:
                buy_signals += 1
            elif df['DMP_14'].iloc[-1] < df['DMN_14'].iloc[-1]:
                sell_signals += 1
        else:
            neutral_signals += 1

        # Williams %R
        if df['WILLR_14'].iloc[-1] > -20:
            sell_signals += 1
        elif df['WILLR_14'].iloc[-1] < -80:
            buy_signals += 1
        else:
            neutral_signals += 1

        # CCI
        if df['CCI_14_0.015'].iloc[-1] > 100:
            sell_signals += 1
        elif df['CCI_14_0.015'].iloc[-1] < -100:
            buy_signals += 1
        else:
            neutral_signals += 1

        # Ultimate Oscillator
        if df['UO_7_14_28'].iloc[-1] > 70:
            sell_signals += 1
        elif df['UO_7_14_28'].iloc[-1] < 30:
            buy_signals += 1
        else:
            neutral_signals += 1

        # ROC
        if df['ROC_14'].iloc[-1] > 0:
            buy_signals += 1
        elif df['ROC_14'].iloc[-1] < 0:
            sell_signals += 1
        else:
            neutral_signals += 1

        # Parabolic SAR (used as Bull/Bear Power)
        if df['Close'].iloc[-1] > df['PSARl_0.02_0.2'].iloc[-1]:
            buy_signals += 1
        elif df['Close'].iloc[-1] < df['PSARs_0.02_0.2'].iloc[-1]:
            sell_signals += 1
        else:
            neutral_signals += 1

        current_atr = df['ATRr_14'].iloc[-1]
        historical_atr = df['ATRr_14'].mean()
        atr_threshold_low = 0.5 * historical_atr
        atr_threshold_high = 1.5 * historical_atr
        volatility_signal = "Neutral"
        if current_atr < atr_threshold_low:
            volatility_signal = "Less Volatility"
        elif current_atr > atr_threshold_high:
            volatility_signal = "High Volatility"
        
        decision = {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'neutral_signals': neutral_signals,
            'volatility': volatility_signal
        }
        return decision

    def sma_signals(df, price):
        signals = {}
        sma_periods = [5, 10, 20, 50, 100, 200]

        for period in sma_periods:
            sma_column = f'SMA_{period}'
            if sma_column in df.columns:
                if price > df[sma_column].iloc[-1]:
                    signals[sma_column] = 'Buy'
                else:
                    signals[sma_column] = 'Sell'

        buy_count = sum(signal == 'Buy' for signal in signals.values())
        sell_count = sum(signal == 'Sell' for signal in signals.values())
        summary_signal = "BUY" if buy_count > sell_count else "SELL" if sell_count > buy_count else "NEUTRAL"

        return {
            "signals": signals,
            "summary": summary_signal,
            "buy_count": buy_count,
            'sell_count': sell_count
        }

    current_price = df['Close'].iloc[-1]
    sma_decision = sma_signals(df, current_price)

    print("SMA Signals:")
    for key, value in sma_decision['signals'].items():
        print(f"{key}: {value}")

    df.ta.strategy(CustomStrategy)

    decision = make_decision(df)

    print("\nIndicators:")
    print(f"Buy - {decision['buy_signals']}")
    print(f"Sell - {decision['sell_signals']}")
    print(f"Neutral - {decision['neutral_signals']}")

    def overall_decision(df, price):
        indicator_decision = make_decision(df)
        sma_decision = sma_signals(df, price)

        combined_buy_signals = indicator_decision['buy_signals'] + sma_decision['buy_count']
        combined_sell_signals = indicator_decision['sell_signals'] + sma_decision['sell_count']
        combined_neutral_signals = indicator_decision['neutral_signals']

        total_combined_signals = combined_buy_signals + combined_sell_signals + combined_neutral_signals
        combined_buy_ratio = combined_buy_signals / total_combined_signals if total_combined_signals else 0
        combined_sell_ratio = combined_sell_signals / total_combined_signals if total_combined_signals else 0
        combined_neutural_ratio = combined_neutral_signals / total_combined_signals if total_combined_signals else 0

        if combined_buy_ratio > 0.7:
            final_decision = "STRONG BUY"
        elif combined_buy_ratio > 0.55:
            final_decision = "BUY"
        elif combined_sell_ratio > 0.7:
            final_decision = "STRONG SELL"
        elif combined_sell_ratio > 0.55:
            final_decision = "SELL"
        elif combined_neutural_ratio > 0.6:
            final_decision = "HOLD"
        elif combined_buy_signals > combined_sell_signals:
            final_decision = "BUY"
        elif combined_sell_signals > combined_buy_signals:
            final_decision = "SELL"
        else:
            final_decision = "HOLD"

        volatility_signal = indicator_decision['volatility']
        return f"{final_decision}, volatility: {volatility_signal}"

    final_decision = overall_decision(df, current_price)
    print(f"\nFinal decision based on the SMA and indicators is: {final_decision}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        main(ticker)
    else:
        print("Please provide a ticker symbol.")