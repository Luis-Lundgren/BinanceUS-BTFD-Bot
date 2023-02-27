"""
A python script with different utilities
Original Code by Cedric Holz ( https://github.com/cedricholz/Twitter-Crypto-Signal-Binance-Bot )
Modified by Tru3Nrg
"""

import os
from datetime import datetime
import urllib
import json
import time
from datetime import date
import binance_utils
from decimal import *
# import ta
import csv
import pandas as pd
import numpy as np
from numpy import genfromtxt
from pathlib import Path
import discord_notify as dn


def get_set_from_file(filename):
    with open(filename, 'r') as f:
        file_contents = json.loads(f.read())

    return set(file_contents)


def get_dict_from_file(filename):
    with open(filename, 'r') as f:
        file_contents = json.loads(f.read())
    return file_contents


def save_dict_to_file(filename, dict):
    j = json.dumps(dict)
    f = open(filename, 'w')
    f.write(j)
    f.close()


def get_date_time():
    now = datetime.now()
    return "%s:%s:%s %s/%s/%s" % (now.hour, now.minute, now.second, now.month, now.day, now.year)


def print_and_write_to_logfile(log_text):
    timestamp = '[' + get_date_time() + '] '
    log_text = timestamp + log_text
    if log_text is not None:
        print(log_text)

        with open('logs.txt', 'a') as myfile:
            myfile.write(log_text + '\n')


def print_and_write_spent_budget(spent_budget):
    # timestamp = '[' + get_date_time() + '] '
    log_text = spent_budget
    if log_text is not None:
        print(log_text)

        with open('spent_budget.txt', 'a') as myfile:
            myfile.write(str(log_text) + '\n')


def print_and_write_to_logfile_test(log_text):
    timestamp = '[' + get_date_time() + '] '
    log_text = timestamp + log_text
    if log_text is not None:
        print(log_text)

        with open('test_log.txt', 'a') as myfile:
            myfile.write(log_text + '\n')


def percent_change(bought_price, cur_price):
    if bought_price == 0:
        return 0

    return 100 * (cur_price - bought_price) / bought_price


def create_market_directory(symbol):
    market_folder_path = 'DataFrames/{}'.format(symbol)
    if not os.path.exists(market_folder_path):
        os.mkdir(market_folder_path)


def save_features_to_dataframe(symbol, df):
    features_df = df.copy()
    df_folder_path = Path('DataFrames/{}/Features'.format(symbol))
    file_to_save = df_folder_path / '{}_features.csv'.format(symbol)
    features_df.to_csv(file_to_save)
    print('FEATURES DATAFRAME SAVED')


def purchase_record(symbol, trades_budget, bought_price, amount):
    time.sleep(2)
    df_folder = Path('DataFrames/{}/Features'.format(symbol))
    trades_folder_path = 'DataFrames/{}/Trades/'.format(symbol)
    if not os.path.exists(trades_folder_path):
        os.mkdir(trades_folder_path)
    file_to_open = df_folder / '{}_features.csv'.format(symbol)
    df = pd.read_csv(file_to_open)
    # print(df)
    purchase_df = df.tail(1).copy()
    # purchase_df = df.head(1).copy() # test only
    # print(purchase_df)
    purchase_df['Buy_Price'] = bought_price
    bnb_fee = binance_utils.binance_fee(trades_budget)
    purchase_df['Binance_Fee'] = '%.8f' % bnb_fee
    purchase_df['Amount_Bought'] = Decimal(amount)
    # print(purchase_df)
    trades_folder_path_symbol = 'DataFrames/{}/Trades/'.format(symbol)
    if not os.path.exists(trades_folder_path_symbol):
        os.mkdir(trades_folder_path_symbol)
    purchase_df_folder = Path(trades_folder_path_symbol)
    file_to_save = purchase_df_folder / '{}_buy_orders.csv'.format(symbol)
    if file_to_save.is_file():
        # print(f'The file {file_to_save} exists')
        purchase_df.to_csv(file_to_save, mode='a', index=False, header=False)
    else:
        # print(f'The file {file_to_save} does not exist')
        purchase_df.to_csv(file_to_save, index=False)


def signal_record(ns_df, symbol):
    signals_folder_path_symbol = 'DataFrames/{}/Signals/'.format(symbol)
    if not os.path.exists(signals_folder_path_symbol):
        os.mkdir(signals_folder_path_symbol)
    purchase_df_folder = Path(signals_folder_path_symbol)
    file_to_save = purchase_df_folder / '{}_signals.csv'.format(symbol)
    if file_to_save.is_file():
        # print(f'The file {file_to_save} exists')
        ns_df.to_csv(file_to_save, mode='a', index=True, header=False)
    else:
        # print(f'The file {file_to_save} does not exist')
        ns_df.to_csv(file_to_save, index=True)


def spent_budget_record(symbol, df, trades_budget):
    spent_budget_folder_path_symbol = 'DataFrames/{}/Budget/'.format(symbol)
    if not os.path.exists(spent_budget_folder_path_symbol):
        os.mkdir(spent_budget_folder_path_symbol)
    spent_budget_df = df.tail(1).copy()
    print(spent_budget_df)
    # spent_budget_df['Yearly_Budget'] = yearly_budget
    spent_budget_df['Trades_Budget'] = trades_budget
    spent_budget_df_folder = Path(spent_budget_folder_path_symbol)
    file_to_save = spent_budget_df_folder / '{}_spent_budget.csv'.format(symbol)
    if file_to_save.is_file():
        # print(f'The file {file_to_save} exists')
        spent_budget_df.to_csv(file_to_save, mode='a', index=True, header=False)
    else:
        # print(f'The file {file_to_save} does not exist')
        spent_budget_df.to_csv(file_to_save, index=True)


def check_budget(symbol):
    spent_budget_df_folder = Path('DataFrames/{}/Budget/'.format(symbol))
    file_to_open = spent_budget_df_folder / '{}_spent_budget.csv'.format(symbol)
    if file_to_open.is_file():
        df = pd.read_csv(file_to_open)
        check_budget_df = df[['Trades_Budget']].copy()
        check_budget_df['Trades_Budget'] = check_budget_df['Trades_Budget'].astype(float)
        spent_budget_total = check_budget_df['Trades_Budget'].sum()
        print("Total USD Amount Spent: {}".format(spent_budget_total))
        return spent_budget_total
    else:
        return None


def discord_notification(notification_text):
    timestamp = '[' + get_date_time() + '] '
    notification_text = timestamp + notification_text
    # Check out https://pypi.org/project/discord-notify/ for instructions

    url = 'your_channel_webhook_url'

    notifier = dn.Notifier(url)

    notifier.send(notification_text, print_message=False)
