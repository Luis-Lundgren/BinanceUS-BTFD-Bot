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


import sys
import traceback
import numpy as np
import talib
from binance_utils import Binance
import utils
import time
import time_sync
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import pandas_ta as ta
from pathlib import Path

binance = Binance()


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
    raw_data = binance.get_daily_historical_candles_binance(symbol, interval,
                                                            start_date,
                                                            end_date)
    # raw_data = binance_utils.get_daily_historical_candles_binance(symbol, interval,
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

    newdf = orig_dataset[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
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
    ti['ema_5'] = ti.Close.ewm(span=5).mean().fillna(0)
    ti['ema_150'] = ti.Close.ewm(span=150).mean().fillna(0)
    ti['rsi'] = talib.RSI(ti.Close.values, timeperiod=14)
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
    # computes features to define target buy signal
    ti['belowEMA005'] = np.where(ti['Close'] < ti['ema_5'], 1, 0)
    ti['belowEMA150'] = np.where(ti['Close'] < ti['ema_150'], 1, 0)
    ti['STK_Oversold'] = np.where(
        (ti.STOCHk_14_3_3 < 15) & (ti.STOCHd_14_3_3 < 20) & (ti.STOCHk_14_3_3 > ti.STOCHd_14_3_3) & (
                ti.STOCHk_14_3_3.shift(1) < ti.STOCHd_14_3_3.shift(1)), 1, 0)
    # ti['oversoldRSI'] = np.where(ti['rsi'] < 30, 1, 0)
    # ti['overboughtRSI'] = np.where(ti['rsi'] > 70, 1, 0)
    ti['rsi_val'] = np.where((ti.rsi < 50), 1, 0)
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


def buy_signal_validation(symbol, data_ti):
    usd_amount = trades_budget
    last_Buy_Signal = data_ti['Buy_Signal'].values[-1]
    s_df = data_ti.tail(1).copy()
    # last_Buy_Signal = 1  # Testing Only
    if last_Buy_Signal == 1:
        mlog("{} IS READY TO BUY".format(symbol))
        # bought, bought_price, amount = binance.market_buy_from_binance_USD(symbol,
        #                                                                    market_max_buy_percent_from_first_order,
        #                                                                    usd_amount)
        # TESTING ONLY: comment below and uncomment above when ready to use real money
        bought, bought_price, amount = binance.test_market_buy_from_binance_USD(symbol,
                                                                                      market_max_buy_percent_from_first_order,
                                                                                      usd_amount)

        if bought:
            print("BUY SUCCESS: {}, {}, {}".format(bought, bought_price, amount))
            # Buy Order Record (optional)
            # utils.purchase_record(symbol,trades_budget, bought_price, amount) # uncomment for record keeping
            # Spent Budget Record
            utils.spent_budget_record(symbol, s_df, trades_budget)  # stores the spent usd amount in a csv
            return True
        else:
            print("CANCELED MARKET ORDER FOR {}".format(symbol))
            return False

    else:
        mlog("NO BUY SIGNAL FOR {}".format(symbol))
        # Signal Record
        # utils.signal_record(s_df, symbol)  # uncomment for record keeping
        print(s_df)  # prints the signals
        return False


def do(market):
    spent_budget = utils.check_budget(market)
    if not spent_budget:
        pass
    else:
        if spent_budget != yearly_principal:
            pass
        else:
            mlog("BUDGET SPENT: BTFD BOT STOPPING")
            # utils.discord_notification("BUDGET SPENT: BTFD BOT STOPPING") # uncomment to notify via Discord
            sys.exit("BTFD BOT STOPPING")

    time_sync.run()  # Windows Only: make sure to run Pycharm as administrator
    date_start_utc, date_end_utc = get_start_and_end_dates()
    utils.create_market_directory(market)
    newdf = data_setup(market, str(date_start_utc),
                       str(date_end_utc))  # get raw data from Binance and return dataframe
    ti = ti_features(newdf)  # calculate and store technical indicators
    ti = compute_features(ti)  # define and store conditions for unique indicators
    ti = define_target_conditions(ti)  # define and store buy signal target conditions
    # Save dataframe (optional)
    # utils.save_features_to_dataframe(market, ti) # uncomment to keep record
    dip_is_bought = buy_signal_validation(market, ti)
    while not dip_is_bought:
        sleep_min, sleep_date = time_sync.sleep_till_time()  # checks every four (4) hours
        mlog("NO DIP TO BUY: GOING TO SLEEP TILL {}".format(sleep_date))

        if dip_is_bought:
            continue

        time.sleep(60 * sleep_min)
        # time.sleep(5)  # Testing Only
        do(market)
    sleep_till_next_check_date, next_check_date = time_sync.sleep_for_long_time()  # checks again in forty eight (48) hours
    mlog("BOUGHT THE F* DIP: GOING TO SLEEP TILL {}".format(next_check_date))
    time.sleep(60 * sleep_till_next_check_date)
    do(market)


def run(market):
    try:
        spent_budget = utils.check_budget(market)
        if not spent_budget:
            do(market)
        else:
            if spent_budget != yearly_principal:
                do(market)
            else:
                mlog("BUDGET SPENT: BTFD BOT STOPPING")
                # utils.discord_notification("BUDGET SPENT: BTFD BOT STOPPING") # uncomment to notify via Discord
                sys.exit("BTFD BOT STOPPING")
    except Exception as e:
        utils.print_and_write_to_logfile_test(traceback.format_exc())
        utils.discord_notification("Something Went Wrong...")
        sleep_min, sleep_date = time_sync.sleep_till_time()
        utils.print_and_write_to_logfile_test("Going to sleep till {} before trying again...".format(sleep_date))
        time.sleep(60 * sleep_min)


if __name__ == '__main__':
    '''
    Script to automate buying bitcoin dips with backtesting
    Principal/Yearly Budget = $1200 a year
    Monthly budget = $100
    Backtest using hourly and daily
    Technical Indicators = Stoch, RSI, MACD, EMA 5, EMA 150
    RSI < 50 OR < 35
    Close < EMA 5 & Close < EMA 150
    MACD < Signal 
    '''

    # Display options for properly viewing Panda Dataframes. Feel Free to comment.
    pd.options.display.width = None
    pd.options.display.max_columns = None
    pd.set_option('display.max_rows', 3000)
    pd.set_option('display.max_columns', 3000)

    market = 'BTCUSD'

    yearly_principal = 12000  # this assumes $1000 per month budget
    number_of_trades = 3  # use the backtesting_binance_btfd.py script to determine the number of yearly trades
    global trades_budget
    trades_budget = yearly_principal / number_of_trades
    global market_max_buy_percent_from_first_order
    # 0.1 = 1%
    market_max_buy_percent_from_first_order = 0.1
    global spent_budget
    spent_budget = 0
    global interval
    interval = binance.data_client.KLINE_INTERVAL_1DAY  # Daily Candles Data
    # interval = binance.data_client.KLINE_INTERVAL_1HOUR # Hourly Candles Data

    run(market)
