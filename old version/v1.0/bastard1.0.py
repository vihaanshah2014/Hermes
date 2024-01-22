# B.A.S.T.A.R.D - Boldly Analyzing and Strategizing Trade Algorithm for Radical Dividends


from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime

# news api
from alpaca_trade_api import REST
from timedelta import Timedelta

# importing AI
from finbert_utils import estimate_sentiment


API_KEY = "PKPQ89WRWLU05IA5JK52"
API_SECRET = "4eKdigiTzaQNoKsgPnG6G5dlVjzAz91f9dKxTZBJ"
BASE_URL = "https://paper-api.alpaca.markets"

ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}

class MLTrader(Strategy):
    def initialize(self, symbol:str="SPY", cash_at_risk:float=.5):
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)


    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price)
        return cash, last_price, quantity
    
    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    
    def get_sentiment(self):
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)

        news = [ev.__dict__["_raw"]["headline"] for ev in news]

        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment


    def risk_values(self, buy=True):
        cash = self.get_cash()
        if buy:
            min_value = 0.95
            max_value = 1.20
        else:
            min_value = 0.8
            max_value = 1.05

        return min_value, max_value
    
    def get_probability_value(self):
        base_cash = 100000  # Set the base cash value
        cash_reserve_percentage = self.get_cash() / base_cash

        # Customize this formula based on your risk strategy
        probability_value = 0.8 + 0.05 * cash_reserve_percentage

        # Ensure the probability value is within a reasonable range
        probability_value = max(0.7, min(0.9, probability_value))

        return probability_value





    def on_trading_iteration(self):
        CASH_MINIMUM = 1000  
        probability_value = self.get_probability_value()
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()
        if cash > last_price and cash > CASH_MINIMUM:
            if sentiment == "positive" and probability > probability_value:
                min_val, max_val = self.risk_values(buy=True)
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * max_val,
                    stop_loss_price=last_price * min_val
                )
                self.submit_order(order)
                self.last_trade = "buy"
            elif sentiment == "negative" and probability > probability_value:
                min_val, max_val = self.risk_values(buy=False)
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * min_val,
                    stop_loss_price=last_price * max_val
                )
                self.submit_order(order)
                self.last_trade = "sell"

start_date = datetime(2022,12,31)
end_date = datetime(2023,12,31)
broker = Alpaca(ALPACA_CREDS)

strategy = MLTrader(name='bastardv1', 
                    broker = broker,
                    parameters={"symbol":"SPY",
                    "cash_at_risk":.5})
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol":"SPY"}
)