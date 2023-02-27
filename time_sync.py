import time
from time import ctime
from datetime import datetime
from datetime import timedelta
import binance_utils
import win32api


def run():
    binance = binance_utils.get_binance_account()
    gt = binance.get_server_time()
    tt = time.gmtime(int((gt["serverTime"]) / 1000))
    win32api.SetSystemTime(tt[0], tt[1], 0, tt[2], tt[3], tt[4], tt[5], 0)
    print("TIME SYNCED")


def sleep_till_time():
    now = datetime.now()
    time_now = now.strftime('%Y-%m-%d %H:%M:%S')
    next_hour = datetime.strptime(time_now, '%Y-%m-%d %H:%M:%S') + timedelta(hours=12)
    # print(next_hour)
    hour = next_hour.hour
    # print(hour)
    time_till_sleep = next_hour.strftime(
        '%Y-%m-%d {}:56:00'.format(hour))  # checking before the end of the hour to get the most accurate signals
    # print(time_till_sleep)
    diff = datetime.strptime(time_till_sleep, '%Y-%m-%d %H:%M:%S') - now
    sleep_min = round(diff.total_seconds() / 60)
    # print(sleep_min)
    return sleep_min, time_till_sleep

# test_sleep_time, test_date = sleep_till_time()
# print(test_date)

def sleep_for_long_time():
    now = datetime.now()
    time_now = now.strftime('%Y-%m-%d %H:%M:%S')
    next_check_date = datetime.strptime(time_now, '%Y-%m-%d %H:%M:%S') + timedelta(hours=48)
    # print(next_check_date)
    # day = next_check_date.hour
    # print(day)
    time_till_sleep = next_check_date.strftime('{}'.format(next_check_date))
    # print(time_till_sleep)
    diff = datetime.strptime(time_till_sleep, '%Y-%m-%d %H:%M:%S') - now
    sleep_min = round(diff.total_seconds() / 60)
    # print(sleep_min)
    return sleep_min, next_check_date


# sleep_till_next_check_date, next_check_date = sleep_for_24hrs()
# print(sleep_till_next_check_date, next_check_date)
