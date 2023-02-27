"""
Disclaimer

All investment strategies and investments involve risk of loss.
Nothing contained in this program, scripts, code or repositoy should be
construed as investment advice.Any reference to an investment's past or
potential performance is not, and should not be construed as, a recommendation
or as a guarantee of any specific outcome or profit.

By using this program you accept all liabilities,
and that no claims can be made against the developers,
or others connected with the program.
"""


import os
import numpy as np
from decimal import *
import talib
from binance_utils import Binance
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt, rcParams
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

'''
Script to backtest buying bitcoin dips
Principal = $1200 a year
Monthly budget = $100
Backtest using hourly and daily
Technical Indicators = Stoch, RSI, MACD, EMA 5, EMA 150
Graph BTC candles (OHLC) + Stoch K & D
RSI < 50 OR < 35
Close < EMA 5 & Close < EMA 150
MACD < Signal 
'''

client = Binance()

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)


yearly_principal = 1200  # this assumes $100 per month budget
number_of_trades = 3  # use backtesting to determine the number of yearly trades
global trades_budget
trades_budget = yearly_principal / number_of_trades
global market_max_buy_percent_from_first_order
# 0.1 = 1%
market_max_buy_percent_from_first_order = 0.1


def get_start_and_end_dates():
    utc_now = datetime.utcnow()
    date_end_utc = utc_now.strftime('%Y-%m-%d %H:%M:%S')
    print(date_end_utc)
    # date_start_utc = date_end_utc - timedelta(days=30)
    # date_start_utc = utc_now - timedelta(days=30)
    date_start_utc = utc_now - relativedelta(months=2)
    print(date_start_utc.strftime('%Y-%m-%d %H:%M:%S'))
    return date_start_utc, date_end_utc


def mlog(*text):
    text = [str(i) for i in text]
    text = " ".join(text)

    datestamp = str(datetime.now().strftime("%m%d%y %H:%M:%S"))

    print("[{}] - {}".format(datestamp, text))


def data_setup(symbol, start_date, end_date):
    raw_data = client.get_daily_historical_candles_binance(symbol, client.data_client.KLINE_INTERVAL_1DAY,
                                                                  start_date,
                                                                  end_date)
    # raw_data = binance_utils.get_daily_historical_candles_binance(symbol, client.KLINE_INTERVAL_1HOUR,
    #                                                               start_date,
    #                                                               end_date)
    data_folder = Path("Binance_Data/")
    file_to_open = data_folder / '{}_raw_data.csv'.format(symbol)
    orig_dataset = pd.read_csv(file_to_open, header=None, parse_dates=True,
                               usecols=[*range(0, 6)])
    orig_dataset.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    # print(orig_dataset)

    orig_dataset['Date'] = orig_dataset['Date'].map(lambda x: str(x)[:-3])
    # print(orig_dataset['Date'])
    orig_dataset['Date'] = orig_dataset['Date'].map(
        lambda x: datetime.fromtimestamp(int(x)).strftime('%m/%d/%Y %H:%M:%S'))  # %H:%M

    close = orig_dataset['Close'].values
    open_ = orig_dataset['Open'].values
    high = orig_dataset['High'].values
    low = orig_dataset['Low'].values
    # volume = orig_dataset['Volume'].values

    newdf = orig_dataset[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    # newdf = orig_dataset[['Date', 'Close']]
    # newdf = newdf.fillna(method='ffill')
    # newdf.ta.stoch(high='High', low='Low', k=14, d=3, append=True)
    newdf = newdf.set_index("Date", drop=True)
    newdf.index = pd.to_datetime(newdf.index)
    newdf = newdf.dropna()

    return newdf


# newdf = data_setup(market, str(date_start_utc), str(date_end_utc))


# print(newdf)
# print(newdf.describe())

def ti_features(sdf):
    ti = sdf.copy()
    ti.ta.stoch(high='High', low='Low', k=14, d=3, append=True)
    # Below are extra Technical Indicators (T.I.) to play with
    # SMA
    # ti['SMA_07'] = (sum(ti.Close, 7)) / 7
    # ti['SMA_10'] = (sum(ti.Close, 10)) / 10
    # ti['SMA_20'] = (sum(ti.Close, 20)) / 20
    # ti['SMA_50'] = (sum(ti.Close, 50)) / 50
    # ti['SMA_100'] = (sum(ti.Close, 100)) / 100
    # ti['SMA_200'] = (sum(ti.Close, 200)) / 200
    # EMA
    ti['ema_5'] = ti.Close.ewm(span=5).mean().fillna(0)
    # ti['ema_10'] = ti.Close.ewm(span=10).mean().fillna(0)
    # ti['ema_15'] = ti.Close.ewm(span=15).mean().fillna(0)
    # ti['ema_20'] = ti.Close.ewm(span=20).mean().fillna(0)
    # ti['ema_50'] = ti.Close.ewm(span=50).mean().fillna(0)
    # ti['ema_100'] = ti.Close.ewm(span=100).mean().fillna(0)
    ti['ema_150'] = ti.Close.ewm(span=150).mean().fillna(0)
    # ti['ema_200'] = ti.Close.ewm(span=200).mean().fillna(0)
    # Average True Range: ATR measures market volatility.
    # It is typically derived from the 14-day moving average
    # of a series of true range indicators
    # ti['ATR'] = talib.ATR(ti['High'].values,
    #                       ti['Low'].values,
    #                       ti['Close'].values,
    #                       timeperiod=14)
    # Average Directional Index (ADX)
    # ADX indicates the strength of a trend in price time series.
    # It is a combination of the negative and positive directional
    # movements indicators computed over a period of n past days
    # corresponding to the input window length (typically 14 days)
    # ti['ADX'] = talib.ADX(ti.High, ti.Low, ti.Close, timeperiod=14)
    # Bollinger Bands
    # ti['upperBB'], ti['middleBB'], ti['lowerBB'] = talib.BBANDS(ti['Close'].values, timeperiod=20, nbdevup=2,
    #                                                             nbdevdn=2, matype=0)
    # Commodity Channel Index (CCI)
    # CCI is used to determine whether a stock is overbought or oversold.
    # It assesses the relationship between an asset price,
    # its moving average and deviations from that average.
    # CCI = (typical price − ma) / (0.015 * mean deviation)
    # typical price = (high + low + close) / 3
    # p = number of periods (20 commonly used)
    # ma = moving average
    # moving average = typical price / p
    # mean deviation = (typical price — MA) / p
    # tp = (ti['High'] + ti['Low'] + ti['Close']) / 3
    # ma = tp / 20
    # md = (tp - ma) / 20
    # ti['CCI'] = (tp - ma) / (0.015 * md)
    # SAR
    # ti['SAR'] = talib.SAR(ti['High'].values, ti['Low'].values, acceleration=0.02, maximum=0.2)
    # Relative Strength Index (RSI)
    # RSI compares the size of recent gains to recent losses,
    # it is intended to reveal the strength or weakness of
    # a price trend from a range of closing prices over a time period.
    ti['rsi'] = talib.RSI(ti.Close.values, timeperiod=14)
    # ti['normRSI'] = talib.RSI(ti['Close'].values, timeperiod=14) / 100.
    # William’s %R
    # This shows the relationship between the current closing price
    # and the high and low prices over the latest n days
    # equal to the input window length.
    # ti['Williams %R'] = talib.WILLR(ti.High.values,
    #                                 ti.Low.values,
    #                                 ti.Close.values, 14)
    macd_fast = 12
    macd_slow = 26
    nNine = 9

    ti['macd'], ti['macdSignal'], ti['macdHist'] = talib.MACD(
        ti.Close,
        fastperiod=macd_fast,
        slowperiod=macd_slow,
        signalperiod=nNine)
    ti = ti.dropna()
    return ti


# ti = ti_features(newdf)
# print(ti)


def compute_features(ti):
    ti = ti.copy()
    # computes features for forest decisions
    # ti['aboveEMA100'] = np.where(ti['Close'] > ti['ema_100'], 1, 0)
    # ti['aboveEMA150'] = np.where(ti['Close'] > ti['ema_150'], 1, 0)
    ti['belowEMA005'] = np.where(ti['Close'] < ti['ema_5'], 1, 0)
    ti['belowEMA150'] = np.where(ti['Close'] < ti['ema_150'], 1, 0)
    # ti['aboveUpperBB'] = np.where(ti['Close'] > ti['upperBB'], 1, 0)
    # ti['belowLowerBB'] = np.where(ti['Close'] < ti['lowerBB'], 1, 0)

    # ti['aboveSAR'] = np.where(ti['Close'] > ti['SAR'], 1, 0)

    # ti['STK_Oversold'] = np.where(
    #     (ti.STOCHk_14_3_3 < 20) & (ti.STOCHd_14_3_3 < 20) & (ti.STOCHk_14_3_3 < ti.STOCHd_14_3_3), 1, 0)
    # ti['STK_Oversold'] = np.where(
    #     (ti.STOCHk_14_3_3 < 15) & (ti.STOCHd_14_3_3 < 20) & (ti.STOCHk_14_3_3 < ti.STOCHd_14_3_3), 1, 0)
    # ti['STK_Oversold'] = np.where(
    #     (ti.STOCHk_14_3_3 < 15) & (ti.STOCHd_14_3_3 < 20), 1, 0)
    ti['STK_Oversold'] = np.where(
        (ti.STOCHk_14_3_3 < 15) & (ti.STOCHd_14_3_3 < 20) & (ti.STOCHk_14_3_3 > ti.STOCHd_14_3_3) & (
                ti.STOCHk_14_3_3.shift(1) < ti.STOCHd_14_3_3.shift(1)), 1, 0)
    # ti['oversoldRSI'] = np.where(ti['rsi'] < 30, 1, 0)
    # ti['overboughtRSI'] = np.where(ti['rsi'] > 70, 1, 0)
    # ti['rsi_test'] = np.where((ti.rsi > 53), 1, 0)
    ti['rsi_val'] = np.where((ti.rsi < 50), 1, 0)
    # ti['macd_test'] = np.where((ti.macd > ti.macdSignal), 1, 0)
    ti['macd_val'] = np.where((ti.macd < ti.macdSignal), 1, 0)
    # very important - cleanup NaN values
    ti = ti.fillna(0).copy()

    # ti.tail()

    return ti


# ti = compute_features(ti)


# print(ti)

def define_target_conditions(ti):
    ti = ti.copy()
    ti['Buy_Signal'] = np.where(
        (ti.STK_Oversold == 1) & (ti.rsi_val == 1) & (ti.macd_val == 1) & (
                ti.belowEMA005 == 1) & (ti.belowEMA150 == 1), 1, 0)
    ti = ti.fillna(0).copy()
    return ti


def plot_candles_and_ti(newdf, market):
    # Avoid case-sensitive issues for accessing data.
    # Optional if using pandas_ta
    newdf.columns = [x.lower() for x in newdf.columns]
    # print(newdf.tail())
    # Create our primary chart
    # the rows/cols arguments tell plotly we want two figures
    fig = make_subplots(rows=4, cols=1)
    # Create our Candlestick chart with an overlaid price line
    fig.append_trace(
        go.Candlestick(
            x=newdf.index,
            open=newdf['open'],
            high=newdf['high'],
            low=newdf['low'],
            close=newdf['close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            showlegend=False
        ), row=1, col=1  # <------------ upper chart
    )
    # price Line
    fig.add_trace(
        go.Scatter(
            x=newdf.index,
            y=newdf['close'],
            line=dict(color='#e42fe9', width=1),
            name='close',
        ), row=1, col=1  # <------------ upper chart
    )
    # EMA 5
    fig.add_trace(
        go.Scatter(
            x=newdf.index,
            y=newdf['ema_5'],
            line=dict(color='#ffbf80', width=1),
            name='EMA 5',
        ), row=1, col=1  # <------------ upper chart
    )
    # EMA 150
    fig.add_trace(
        go.Scatter(
            x=newdf.index,
            y=newdf['ema_150'],
            line=dict(color='#00cccc', width=1),
            name='ema 150',
        ), row=1, col=1  # <------------ upper chart
    )
    # # zigzag Line
    # fig.add_trace(
    #     go.Scatter(
    #         x=newdf.index,
    #         y=newdf['zigzag'],
    #         line=dict(color='#a5fb2d', width=1),
    #         name='zizag',
    #     ), row=1, col=1  # <------------ upper chart
    # )
    # target Buy
    fig.append_trace(
        go.Scatter(
            # mode="markers",
            x=newdf.index,
            # y=newdf['stk_oversold'] * newdf['close'],
            # y=newdf['target_buy'] * newdf['close'],
            # y=newdf['target_os_buy'] * newdf['close'],
            y=newdf['buy_signal'] * newdf['close'],
            # y=newdf['buy_signal_final'] * newdf['close'],
            mode='markers',
            marker=dict(symbol='arrow-up', color='#2fe9d1', size=8),
            showlegend=True,
            # alpha=0.7,
            # marker='2',
            # markers={'2': 'tri_up'},
            # line=dict(color='#ff9900', width=2),
            name='Buy Signal',
        ), row=1, col=1  # <------------ upper chart
    )
    # Sell Signal
    # fig.append_trace(
    #     go.Scatter(
    #         # mode="markers",
    #         x=newdf.index,
    #         # y=newdf['stk_oversold'] * newdf['close'],
    #         # y=newdf['target_sell'] * newdf['close'],
    #         # y=newdf['sell_signal'] * newdf['close'],
    #         y=newdf['sell_signal_final'] * newdf['close'],
    #         mode='markers',
    #         marker=dict(symbol='arrow-down', color='#e9bb2f', size=8),
    #         showlegend=True,
    #         # alpha=0.7,
    #         # marker='2',
    #         # markers={'2': 'tri_up'},
    #         # line=dict(color='#ff9900', width=2),
    #         name='Sell Signal',
    #     ), row=1, col=1  # <------------ upper chart
    # )
    # Fast Signal (%k)
    fig.append_trace(
        go.Scatter(
            x=newdf.index,
            y=newdf['stochk_14_3_3'],
            line=dict(color='#2fe99a', width=2),
            name='Stoch K',
        ), row=2, col=1  # <------------ lower chart
    )
    # Slow signal (%d)
    fig.append_trace(
        go.Scatter(
            x=newdf.index,
            y=newdf['stochd_14_3_3'],
            line=dict(color='#e9e52f', width=2),
            name='Stoch D'
        ), row=2, col=1  # <------------ lower chart
    )
    # # RSI
    fig.append_trace(
        go.Scatter(
            x=newdf.index,
            y=newdf['rsi'],
            line=dict(color='#e9e52f', width=2),
            name='RSI'
        ), row=3, col=1  # <------------ lower chart
    )
    ## MACD
    # Fast Signal (%k)
    fig.append_trace(
        go.Scatter(
            x=newdf.index,
            y=newdf['macd'],
            line=dict(color='#ffff00', width=2),
            name='macd',
            # showlegend=False,
            legendgroup='2',
        ), row=4, col=1
    )
    # Slow signal (%d)
    fig.append_trace(
        go.Scatter(
            x=newdf.index,
            y=newdf['macdsignal'],
            line=dict(color='#00ffff', width=2),
            # showlegend=False,
            legendgroup='2',
            name='signal'
        ), row=4, col=1
    )
    # Colorize the histogram values
    colors = np.where(newdf['macdhist'] < 0, '#9933ff', '#9933ff')
    # Plot the histogram
    fig.append_trace(
        go.Bar(
            x=newdf.index,
            y=newdf['macdhist'],
            name='histogram',
            marker_color=colors,
        ), row=4, col=1
    )
    # # Extend our y-axis a bit
    # fig.update_yaxes(range=[-10, 110], row=2, col=1)
    # Add upper/lower bounds
    fig.add_hline(y=0, col=1, row=2, line_color="#666", line_width=2)
    fig.add_hline(y=100, col=1, row=2, line_color="#666", line_width=2)
    # Add overbought/oversold
    fig.add_hline(y=20, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
    fig.add_hline(y=80, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
    # # Add upper/lower bounds
    fig.add_hline(y=0, col=1, row=3, line_color="#666", line_width=2)
    fig.add_hline(y=100, col=1, row=3, line_color="#666", line_width=2)
    # # Add overbought/oversold
    # fig.add_hline(y=30, col=1, row=3, line_color='#336699', line_width=2, line_dash='dash')
    # fig.add_hline(y=70, col=1, row=3, line_color='#336699', line_width=2, line_dash='dash')
    # Add overbought/oversold
    fig.add_hline(y=50, col=1, row=3, line_color='#336699', line_width=2, line_dash='dash')

    # Make it pretty
    # layout = go.Layout(
    #     plot_bgcolor='#efefef',
    #     # Font Families
    #     font_family='Monospace',
    #     font_color='#000000',
    #     font_size=20,
    #     xaxis=dict(
    #         rangeslider=dict(
    #             visible=False
    #         )
    #     )
    # )
    # fig.update_layout(layout)
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='white',
            linewidth=2,
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            title_text='Close (USD)',
            titlefont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='white',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
        ),
        showlegend=True,
        template='plotly_dark'

    )

    annotations = []
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                            xanchor='left', yanchor='bottom',
                            text='{} BTFD Plot'.format(market),
                            font=dict(family='Rockwell',
                                      size=26,
                                      color='white'),
                            showarrow=False))
    fig.update_layout(annotations=annotations)
    # View our chart in the system default HTML viewer (Chrome, Firefox, etc.)
    fig.show()


def backtest_hodl_amount(df, symbol):
    bh_df = df.copy()
    yearly_principal = 1200
    number_of_trades = sum(bh_df.Buy_Signal)
    trades_budget = yearly_principal / number_of_trades
    print(trades_budget)
    trades_budget_dec = Decimal(trades_budget)
    bh_df['Buy_Price'] = bh_df['Buy_Signal'] * bh_df['Close']
    nz_df = bh_df[['Buy_Price']].copy()
    nz_df = nz_df[(nz_df.T != 0).any()]
    nz_df['Bought'] = nz_df['Buy_Price']
    for i, j, in nz_df['Buy_Price'].iteritems():
        if j != 0:
            amount = client.backtest_get_binance_amount_to_buy_and_order_rate(symbol, j, trades_budget_dec)
            nz_df.loc[[i], ['Bought']] = amount  # Decimal(amount)
            # print('amount bought : {}'.format(amount))
    nz_df = nz_df.round(4)
    print(nz_df)
    bh_df['Bought'] = nz_df['Bought']
    bh_df = bh_df.fillna(0)
    bh_df['Bought'] = bh_df['Bought'].astype(float)
    bh_df['Bought'] = bh_df['Bought'].round(4)
    # print(bh_df)

    total_amount_bought = bh_df['Bought'].sum()
    # total_amount_bought = total_amount_bought.round(4)
    print("{} Total Amount Bought: {}".format(symbol, total_amount_bought))


market = 'BTCUSD'
date_end = '2021-12-31 16:00:00'
date_end_utc = datetime.strptime(date_end, '%Y-%m-%d %H:%M:%S') + timedelta(hours=8)
print(date_end_utc)
# date_start_utc = date_end_utc - timedelta(days=32)
date_start_utc = date_end_utc - relativedelta(years=1)
print(date_start_utc)

# date_start_utc, date_end_utc = get_start_and_end_dates()

newdf = data_setup(market, str(date_start_utc), str(date_end_utc))
# print(newdf)
ti = ti_features(newdf)
# print(ti)
ti = compute_features(ti)
ti = define_target_conditions(ti)
backtest_hodl_amount(ti, market)  # see when and how much BTC would have been bought
plot_candles_and_ti(ti, market)  # plot BTC candles graph with Technical Indicators
