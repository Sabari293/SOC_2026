import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.signal import lfilter

def EMA(price:np.ndarray,N:int):
    r=(2/(N+1))
    return lfilter([r], [1, -(1-r)], price)


class MACD:
    def __init__(self,price:np.ndarray):
        self.price=price

        self.ema12=EMA(price,12)
        self.ema26=EMA(price,26)
        self.macd=self.ema12-self.ema26
        self.signal=EMA(self.macd,9)

        self.a12=2/(12+1)
        self.a26=2/(26+1)
        self.a9=2/(9+1)

    def emanew(self, current_price):
        self.price = np.append(self.price, current_price)
        
        new_ema12=self.a12*current_price + (1-self.a12)*self.ema12[-1]
        new_ema26=self.a26*current_price + (1-self.a26)*self.ema26[-1]

        macd=new_ema12-new_ema26
        new_signal=self.a9*macd + (1-self.a9)*self.signal[-1]
        self.ema12=np.append(self.ema12, new_ema12)
        self.ema26=np.append(self.ema26, new_ema26)
        self.macd=np.append(self.macd, macd)
        self.signal=np.append(self.signal, new_signal)

        return macd,new_signal
    def get_macd(self):
        return self.macd

class RSI:
    def __init__(self, price: np.ndarray, period: int = 14):
        self.price=price
        self.period=period
        diff=np.diff(self.price)
        gain=np.maximum(diff, 0)
        loss=np.maximum(-diff, 0)
        self.avg_gain=np.mean(gain[:period])
        self.avg_loss=np.mean(loss[:period])
        self.rsi=np.full(len(self.price), np.nan)
        rs=self.avg_gain/self.avg_loss if self.avg_loss != 0 else np.inf
        self.rsi[period] = 100 - 100/(1+rs)

        for i in range(period + 1, len(self.price)):
            change = self.price[i] - self.price[i - 1]
            gain = max(change, 0)
            loss = max(-change, 0)
            self.avg_gain = ((period - 1) * self.avg_gain + gain) / period
            self.avg_loss = ((period - 1) * self.avg_loss + loss) / period

            rs = self.avg_gain / self.avg_loss if self.avg_loss != 0 else np.inf
            self.rsi[i]=100-100/(1+rs)

    def rsinew(self, current_price):
        change=current_price - self.price[-1]
        gain = max(change, 0)
        loss = max(-change, 0)
        self.avg_gain = ((self.period - 1) * self.avg_gain + gain) / self.period
        self.avg_loss = ((self.period - 1) * self.avg_loss + loss) / self.period
        rs = self.avg_gain / self.avg_loss if self.avg_loss != 0 else np.inf
        new_rsi = 100 - 100 / (1 + rs)
        self.price=np.append(self.price,current_price)
        self.rsi=np.append(self.rsi, new_rsi)

        return new_rsi

    def get_rsi(self):
        return self.rsi