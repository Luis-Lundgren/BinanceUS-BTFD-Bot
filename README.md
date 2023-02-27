# Binance Buy The F* Dip Bot

## Description
Binance-BTFD-Bot is a Python Script that automates buying and hodling Bitcoin dips using Binance Data and calculating
Technical Indicators.

The bot checks every 12 hours and using one day candles calculates the technical indicators.

- The bot checks if the technical indicators (i.e. Stoch, RSI, MACD, EMA) of the BTCUSD pair give a buy signal
- The bot will buy the `trades_budget` (i.e. $4000 USD) worth of Bitcoin on Binance.us
- The bot will repeat the above until the `yearly_principal` has been spent

## DISCLAIMER:

**NOT FINANCIAL ADVICE - USE AT YOUR OWN RISK - NOT RESPONSIBLE FOR ANY FINANCIAL LOSSES YOU MAY INCUR**

This software is provided as is with no guarantees. Buying or Investing in Bitcoin involves risk. While it is possible
to minimize risk, your investments are solely your responsibility. It is imperative that you conduct your own research.

## Getting Started with Binance BTFD Bot
You will need a computer, a [binance account](https://www.binance.us/en/home), and a copy of this code.

## Quick Links
 * [Pycharm](https://www.jetbrains.com/pycharm/) **TESTED ON WINDOWS 10 PYCHARM CE 2020.3 AND 2021.1**
* [Python Download (3.7.0)](https://www.python.org/downloads/release/python-370/)
* [Anaconda](https://www.anaconda.com/products/individual)
   



## Installation
1. Download and Install Pycharm
2. Create a [Conda](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html#3038b494) environment 
3. Download the repository
4. Use the package manager in Pycharm to install requirements.
5. To use Ta-Lib follow the instructions [here](https://blog.quantinsti.com/install-ta-lib-python/)


## Usage

1. Open Pycharm as Administrator (WINDOWS ONLY)
2. Install Dependencies
3. Run `backtest_binance_btfd.py` to get familiarity with the indicators and strategy
4. Once confident edit the `binance_secrets.json` and update with your credentials.
5. in `utils.py` make sure to update the [discord notify](https://pypi.org/project/discord-notify/) with your webhook url
6. Run `binance_btfd.py`
7. HODL


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
