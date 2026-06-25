import numpy as np
import matplotlib.pyplot as plt
from indicators import MACD, RSI
from filters import VolumeFilter

    
class SignalGenerator:
    def __init__(self,macd_indicator,rsi_indicator,volume_filter,rsi_buy=30,rsi_sell=70):

        self.macd = macd_indicator
        self.rsi = rsi_indicator
        self.volume = volume_filter
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.position=0

        self.signal = []

    def generatenew(self):
        
        if len(self.macd.macd) < 2:
            self.signal.append(0)
            return 0

        prev_macd = self.macd.macd[-2]
        curr_macd = self.macd.macd[-1]

        prev_signal = self.macd.signal[-2]
        curr_signal = self.macd.signal[-1]

        curr_rsi = self.rsi.rsi[-1]
        volume_ok = self.volume.valid[-1]

        buy = (((prev_macd <= prev_signal and curr_macd > curr_signal) or curr_rsi < self.rsi_buy) and volume_ok)

        sell = (((prev_macd >= prev_signal and curr_macd < curr_signal) or curr_rsi > self.rsi_sell) and volume_ok)
        print("MACD Cross Up: {prev_macd <= prev_signal and curr_macd > curr_signal}, ""RSI: {curr_rsi:.2f}, ""Volume OK: {volume_ok}, ""Buy: {buy}")
        if buy:
            self.signal.append(1)
            return 1

        elif sell:
            self.signal.append(-1)
            return -1

        self.signal.append(0)
        return 0
        

    def signals(self):
        return np.array(self.signal)
    def generatehistory(self):
        self.signal = np.zeros(len(self.macd.macd), dtype=int)

        for i in range(1, len(self.macd.macd)):
            prev_macd = self.macd.macd[i-1]
            curr_macd = self.macd.macd[i]

            prev_signal = self.macd.signal[i-1]
            curr_signal = self.macd.signal[i]

            curr_rsi = self.rsi.rsi[i]
            volume_ok = self.volume.valid[i]

            buy = (((prev_macd <= prev_signal and curr_macd > curr_signal) or curr_rsi < self.rsi_buy) and volume_ok)
            sell = (((prev_macd >= prev_signal and curr_macd < curr_signal) or curr_rsi > self.rsi_sell) and volume_ok)

            if buy:
                self.signal[i] = 1
            elif sell:
                self.signal[i] = -1


class Stock:
    def __init__(self, prices: np.ndarray, volume: np.ndarray):
        self.price = prices
        self.volume = volume

        self.macd = MACD(prices)
        self.rsi = RSI(prices)

        self.volume_filter = VolumeFilter(volume)
        self.signal_generator = SignalGenerator(self.macd,self.rsi,self.volume_filter)

    def update(self, new_price, new_volume):
        self.macd.emanew(new_price)
        self.rsi.rsinew(new_price)
        self.volume_filter.volumenew(new_volume)

        return self.signal_generator.generatenew()
    
    def plot(self):
        fig, ax = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

        
        ax[0].plot(self.price, label="Close Price", color="black")
        ax[0].set_title("Price")
        ax[0].grid(True)
        ax[0].legend()

        signals = np.array(self.signal_generator.signal)

        buy = np.where(signals == 1)[0]
        sell = np.where(signals == -1)[0]

        if len(buy):
            ax[0].scatter(buy, self.price[buy],marker="^", color="green",s=100, label="Buy")

        if len(sell):
            ax[0].scatter(sell, self.price[sell],marker="v", color="red",s=100, label="Sell")

        ax[0].legend()

        ax[1].plot(self.macd.macd, label="MACD")
        ax[1].plot(self.macd.signal, label="Signal")
        ax[1].bar(range(len(self.macd.macd)),self.macd.macd-self.macd.signal,alpha=0.4,label="Histogram")

        ax[1].axhline(0, color="black", lw=0.8)
        ax[1].set_title("MACD")
        ax[1].grid(True)
        ax[1].legend()

        ax[2].plot(self.rsi.rsi, label="RSI", color="purple")
        ax[2].axhline(70, color="red", linestyle="--")
        ax[2].axhline(30, color="green", linestyle="--")
        ax[2].set_ylim(0, 100)
        ax[2].set_title("RSI")
        ax[2].grid(True)
        ax[2].legend()

        ax[3].bar(range(len(self.volume_filter.volume)),self.volume_filter.volume,alpha=0.6,label="Volume")

        ax[3].plot(self.volume_filter.sma,color="orange",linewidth=2,label="Volume SMA")

        ax[3].set_title("Volume")
        ax[3].grid(True)
        ax[3].legend()

        # signals = np.array(self.signal_generator.signal)

        # shares = 0
        # cash = 0.0
        # equity = np.zeros(len(self.price))

        # for i in range(len(self.price)):
        #     if signals[i] == 1:          
        #         shares += 10
        #         cash -= 10 * self.price[i]

        #     elif signals[i] == -1:       
        #         shares -= 10
        #         cash += 10 * self.price[i]

        #     equity[i] = cash + shares * self.price[i]

        # ax[4].plot(equity, color="green", linewidth=2, label="Portfolio Value")
        # ax[4].axhline(0, color="black", linestyle="--", linewidth=1)

        # ax[4].set_title("Portfolio PnL")
        # ax[4].set_ylabel("Value")
        # ax[4].grid(True)
        # ax[4].legend()

        # print("Final Cash      : {cash:.2f}")
        # print("Final Shares    : {shares}")
        # print("Final Portfolio : {equity[-1]:.2f}")

        plt.tight_layout()
        plt.show()