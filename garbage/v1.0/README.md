# B.A.S.T.A.R.D - Boldly Analyzing and Strategizing Trade Algorithm for Radical Dividends

## Overview
B.A.S.T.A.R.D is a trade algorithm designed to make bold decisions by analyzing market sentiment using natural language processing (NLP). It leverages sentiment analysis on financial news headlines to strategically execute trades for potentially radical dividends. The algorithm integrates with Alpaca's trading platform and employs the FinBERT model for sentiment analysis.

## Setup
Before using B.A.S.T.A.R.D, ensure you have the required dependencies and configurations:

### Dependencies
- [lumibot](https://github.com/lumibot/lumibot): A Python library for building trading algorithms.
- [Alpaca API](https://alpaca.markets/): An API for commission-free stock trading.
- [finbert](https://github.com/ProsusAI/finbert): A pre-trained NLP model for financial sentiment analysis.
- [transformers](https://huggingface.co/transformers/): A library for natural language processing models.

### Configuration
Set up your Alpaca API credentials and FinBERT model. Replace the placeholder values in the script with your API key, secret, and model details.

```python
API_KEY = "YOUR_ALPACA_API_KEY"
API_SECRET = "YOUR_ALPACA_API_SECRET"
BASE_URL = "https://paper-api.alpaca.markets"

ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]
```

## Algorithm Components

### MLTrader Class
The `MLTrader` class extends the lumibot `Strategy` class and defines the main trading logic. It integrates sentiment analysis to make trading decisions based on positive or negative market sentiment.

### Sentiment Analysis
Sentiment analysis is performed using the FinBERT model. The `estimate_sentiment` function takes a list of news headlines, analyzes sentiment, and returns the probability and sentiment label.

```python
tensor, sentiment = estimate_sentiment(['markets responded negatively to the news!', 'traders were displeased!'])
print(tensor, sentiment)
```

## How to Run
1. Ensure all dependencies are installed (`lumibot`, `alpaca-trade-api`, `transformers`).
2. Set up your Alpaca API credentials and FinBERT model details.
3. Adjust the trading parameters, such as the symbol and cash-at-risk percentage.
4. Run the script to backtest the algorithm.

```python
python your_script_name.py
```

## Disclaimer
B.A.S.T.A.R.D is a trading algorithm created for educational and experimental purposes. Use it at your own risk, and do not deploy it with real funds without thorough testing and understanding of its behavior. The algorithm's performance may vary, and past results do not guarantee future success.

## Author
[Your Name]

For more information and updates, check the [lumibot GitHub repository](https://github.com/lumibot/lumibot) and [finbert GitHub repository](https://github.com/ProsusAI/finbert).
