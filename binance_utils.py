"""
A python script with different Binance utilities
Utilized Python-Binance ( https://github.com/sammchardy/python-binance )
Original Code by Cedric Holz ( https://github.com/cedricholz/Twitter-Crypto-Signal-Binance-Bot )
Modified by Luis Lundgren

"""

import os
from binance.client import Client
from binance.enums import *
import utils
import json
import math
from decimal import *
import csv
from pathlib import Path


def get_binance_account():
    with open("binance_secrets.json") as secrets_file:
        secrets = json.load(secrets_file)
        secrets_file.close()

    return Client(secrets['key'], secrets['secret'], tld='us')


class Binance:
    def __init__(self):
        self.data_client = Client("", "", tld='us')
        self.binance = get_binance_account()

    def get_daily_historical_candles_binance(self, symbol, interval, start, end):
        # binance = get_binance_account()
        # client = Client("", "", tld='us')  # not for purchasing, gathering data only
        raw_data = self.data_client.get_historical_klines(symbol, interval, start, end)
        data_folder = "Binance_Data/"
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        data_folder_path = Path("Binance_Data/")
        file_to_open = data_folder_path / '{}_raw_data.csv'.format(symbol)
        csvfile = open(file_to_open, 'w', newline='')
        candlestick_writer = csv.writer(csvfile, delimiter=',')

        for candlestick in raw_data:
            candlestick_writer.writerow(candlestick)
        csvfile.close()

    def backtest_get_binance_amount_to_buy_and_order_rate(self, market, buy_price, usd_amount):
        """
        For Back testing only

        :param market:
        :param buy_price:
        :param usd_amount:
        :return: amount_to_buy
        """

        tickers = self.data_client.get_exchange_info()['symbols']

        ticker = [ticker for ticker in tickers if ticker['symbol'] == market][0]

        constraints = ticker['filters'][2]

        minQty = float(constraints['minQty'])
        maxQty = float(constraints['maxQty'])
        stepSize = Decimal(constraints['stepSize'])

        amount_to_buy = Decimal(usd_amount) / Decimal(buy_price)
        # print(Decimal(usd_amount))
        # print(Decimal(buy_price))
        # print(amount_to_buy)

        if minQty < amount_to_buy < maxQty:
            return amount_to_buy
        else:
            return 0

    def get_market_binance_amount_to_buy_and_order_rate(self, market, total_bitcoin,
                                                        market_max_buy_percent_from_first_order):
        """

        Goes through the sell order book and
        takes the price of the first order that
        is selling the full amount that you can buy.
        Returns the price and amount you can buy.

        """

        tickers = self.binance.get_exchange_info()['symbols']

        ticker = [ticker for ticker in tickers if ticker['symbol'] == market][0]

        constraints = ticker['filters'][2]

        minQty = float(constraints['minQty'])
        maxQty = float(constraints['maxQty'])
        stepSize = Decimal(constraints['stepSize'])

        sell_orders = self.binance.get_order_book(symbol=market)['asks']
        # print(sell_orders)

        initial_price = Decimal(sell_orders[0][0])
        print("Initial Price: {}".format(initial_price))

        for order in sell_orders:
            order_rate = Decimal(order[0])
            # order_rate = Decimal(2.0562)
            print("Order Rate: {}".format(order_rate))
            print(utils.percent_change(initial_price, order_rate))
            if utils.percent_change(initial_price, order_rate) < market_max_buy_percent_from_first_order:

                order_quantity = Decimal(order[1])
                print(order_quantity)
                amount_to_buy = Decimal(total_bitcoin) / Decimal(order_rate)
                print(amount_to_buy)
                constrained_amount_to_buy = math.floor((1 / stepSize) * amount_to_buy) * stepSize
                print(constrained_amount_to_buy)
                if amount_to_buy < order_quantity and minQty < constrained_amount_to_buy < maxQty:
                    return constrained_amount_to_buy, order_rate
            else:
                return 0, 0
        return 0, 0

    def binance_fee(self, trades_budget):
        bnb_avg_price = self.data_client.get_avg_price(symbol='BNBUSD')
        bnb_current_price = bnb_avg_price['price']
        print('Current BNB Price: {}'.format(bnb_current_price))
        # bnb_fee = 0.075 / float(bnb_current_price)
        # bnb_fee = 0.075 / float(bnb_current_price)
        bnb_fee = (trades_budget * 0.075 / 100) / float(bnb_current_price)
        print('BNB FEE: %.8f' % bnb_fee)
        return bnb_fee

    def market_buy_from_binance_USD(self, market, market_max_buy_percent_from_first_order, usd_amount):

        amount, order_price = self.get_market_binance_amount_to_buy_and_order_rate(market, usd_amount,
                                                                              market_max_buy_percent_from_first_order)
        print("AMOUNT: {}".format(amount))
        if amount == 0:
            utils.print_and_write_to_logfile("MARKET ORDER DID NOT GO THROUGH")
            return False, order_price, amount

        order = self.binance.order_market_buy(
            symbol=market,
            quantity=amount)

        if order['status'] == 'FILLED':
            utils.print_and_write_to_logfile("SUCCESSFUL ORDER ON BINANCE")
            utils.print_and_write_to_logfile("MARKET: " + market)
            utils.print_and_write_to_logfile("AMOUNT: " + str(amount))
            utils.print_and_write_to_logfile("ORDER PRICE: " + str(order_price))
            utils.print_and_write_to_logfile("TOTAL USD: " + str(usd_amount))
            return True, order_price, amount

        else:
            return False, order_price, amount

    def test_market_buy_from_binance_USD(self, market, market_max_buy_percent_from_first_order, usd_amount):

        amount, order_price = self.get_market_binance_amount_to_buy_and_order_rate(market, usd_amount,
                                                                              market_max_buy_percent_from_first_order)
        print("AMOUNT: {}".format(amount))
        if amount == 0:
            utils.print_and_write_to_logfile_test("MARKET ORDER DID NOT GO THROUGH")
            return False, order_price, amount

        order = self.binance.create_test_order(
            symbol=market,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            # timeInForce=TIME_IN_FORCE_GTC,
            quantity=amount)
        # price=order_price)

        if order == {}:
            utils.print_and_write_to_logfile_test("Test SUCCESSFUL ORDER ON BINANCE")
            utils.print_and_write_to_logfile_test("Test MARKET: " + market)
            utils.print_and_write_to_logfile_test("Test AMOUNT: " + str(amount))
            utils.print_and_write_to_logfile_test("Test ORDER PRICE: " + str(order_price))
            utils.print_and_write_to_logfile_test("Test TOTAL USD: " + str(usd_amount))
            return True, order_price, amount

        else:
            return False, order_price, amount
