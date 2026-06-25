import numpy as np
from scipy.io import loadmat

from signal_generator import Stock

data = loadmat("inputData_ETF.mat")

syms = [s[0] for s in data["syms"].squeeze()]
cl = data["cl"]
vol = data["vol"]      

idx = syms.index("EWA")

price = cl[:, idx]
volume = vol[:, idx]

mask = ~np.isnan(price) & ~np.isnan(volume)

price = price[mask]
volume = volume[mask]

stock = Stock(price, volume)

stock.signal_generator.generatehistory()

stock.plot()
signals = np.array(stock.signal_generator.signal)

print("Number of Buy signals :", np.sum(signals == 1))
print("Number of Sell signals:", np.sum(signals == -1))
print("Number of Hold signals:", np.sum(signals == 0))