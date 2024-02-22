import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def smma(series, n):
    output = [series[0]]
    for i in range(1, len(series)):
        temp = output[-1] * (n - 1) + series[i]
        output.append(temp / n)
    return output

def rsi(data, n=14):
    delta = data.diff().dropna()
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    rs = np.divide(smma(up, n), smma(down, n))
    output = 100 - 100 / (1 + rs)
    return output[n - 1:]

def signal_generation(df, method, n=14):
    df['rsi'] = 0.0
    df['rsi'][n:] = method(df['Close'], n=14)
    df['positions'] = np.select([df['rsi'] < 30, df['rsi'] > 70], [1, -1], default=0)
    df['signals'] = df['positions'].diff()
    return df[n:]

def plot(new, ticker):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(211)
    new['Close'].plot(label=ticker)
    ax.plot(new.loc[new['signals'] == 1].index, new['Close'][new['signals'] == 1], label='LONG', lw=0, marker='^', c='g')
    ax.plot(new.loc[new['signals'] == -1].index, new['Close'][new['signals'] == -1], label='SHORT', lw=0, marker='v', c='r')
    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Positions')
    plt.xlabel('Date')
    plt.ylabel('price')
    plt.show()
    
    bx = plt.figure(figsize=(10, 10)).add_subplot(212, sharex=ax)
    new['rsi'].plot(label='relative strength index', c='#522e75')
    bx.fill_between(new.index, 30, 70, alpha=0.5, color='#f22f08')
    bx.text(new.index[-45], 75, 'overbought', color='#594346', size=12.5)
    bx.text(new.index[-45], 25, 'oversold', color='#594346', size=12.5)
    plt.xlabel('Date')
    plt.ylabel('value')
    plt.title('RSI')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def main():
    ticker = 'SPY'
    startdate = '2016-01-01'
    enddate = '2018-01-01'
    df = yf.download(ticker, start=startdate, end=enddate)
    new = signal_generation(df, rsi, n=14)
    plot(new, ticker)

if __name__ == '__main__':
    main()
