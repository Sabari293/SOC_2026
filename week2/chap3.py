import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import VarianceRatio
from hurst import compute_Hc
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.io import loadmat
from arch.unitroot.cointegration import engle_granger
from statsmodels.tsa.vector_ar.vecm import coint_johansen



def moving_average(price,window):
    w=np.ones(window)/window
    return np.convolve(price,w,mode='valid')


def moving_std(price,window,ma):
    x_sq=moving_average(price**2,window)
    return np.sqrt(x_sq-ma**2)

def print_resultcj(result):
    print("Trace statistics:")
    print(result.lr1)
    print("Critical values:")
    print(result.cvt)
    print("Eigen statistics:")
    print(result.lr2)
    print("Critical values :")
    print(result.cvm)
    print("Eigen values :")
    print(result.evec)

def fillMissingData(prices):
    prices = prices.copy()

    for t in range(1, len(prices)):
        if not np.isfinite(prices[t]):
            prices[t] = prices[t-1]

    return prices
if __name__=='__main__':
    data = loadmat('inputData_ETF.mat')

    tday = data['tday']
    syms = data['syms']
    cl = data['cl']

    syms = [s[0] for s in syms.squeeze()]

    idxA = np.where(np.array(syms) == 'GLD')[0][0]
    idxB = np.where(np.array(syms) == 'USO')[0][0]

    yA=cl[:,idxA]
    yB=cl[:,idxB]

    result=coint_johansen(np.column_stack((yA,yB)),0,1)
    print_resultcj(result)
    # not cointegrated

    lookback=20
    h=np.zeros_like(yA)
    # print(yA.shape)
    # raise "Errr"
    for t in range(lookback,np.shape(yA)[0]):
        m=LinearRegression()
        m.fit(yA[t-lookback+1:t+1].reshape(-1,1),(yB[t-lookback+1:t+1]))
        h[t]=m.coef_[0]
    y=h*yA-yB
    # plt.plot(yB/yA)
    # plt.show()
    # plt.clf()
    plt.plot(y)
    plt.savefig('spread_uso_gld.png')



    y2=np.column_stack((yA,yB))
    ma1=moving_average(y,lookback)
    std1=moving_std(y,lookback,ma1)
    nUnits1=-(y[lookback-1:]-ma1)/std1
    pos1=(nUnits1[:,None]*np.column_stack((h[lookback-1:],-np.ones_like(h[lookback-1:])))*y2[lookback-1:])
    pos1_lag=pos1[:-1]
    y2_lag=y2[lookback-1:][:-1]
    y2_curr=y2[lookback-1:][1:]
    pnl=np.sum(pos1_lag*(y2_curr-y2_lag)/y2_lag,axis=1)
    ret=pnl/(np.sum(np.abs(pos1_lag),axis=1))
    cp=(np.cumprod(1+ret)-1)
    plt.clf()
    plt.plot(cp)
    plt.savefig("cumret1.png")

    #with ratios

    ratio=yB/yA
    
    yB=yB[lookback-1:]
    yA=yA[lookback-1:]
    ma2=moving_average(ratio,lookback)
    std2=moving_std(ratio,lookback,ma2)
    ratio=ratio[lookback-1:]
    nUnits2=-(ratio-ma2)/std2
    pos2=nUnits2[:,None]*np.ones((yB.shape[0],2))
    pos2[:,1]*=-1
    pos2_lag=pos2[:-1]
    pnl2=np.sum(pos2_lag*(y2_curr-y2_lag)/y2_lag,axis=1)
    ret2=pnl2/np.sum(np.abs(pos2_lag),axis=1)
    plt.clf()
    plt.plot((np.cumprod(ret2+1)-1))
    plt.savefig('ratios_returns.png')

    #Doug bollinger
    entryZ=1
    exitZ=0
    r=np.column_stack((-h[lookback-1:], np.ones_like(h[lookback-1:])))
    yport = np.sum(r * y2[lookback-1:],axis=1)   
    ma3=moving_average(yport,lookback)
    std3=moving_std(yport,lookback,ma3)
    zscore = np.full_like(yport, np.nan, dtype=float)
    zscore[lookback-1:] = (yport[lookback-1:] - ma3)/std3
    long_entry=zscore<-entryZ
    long_exit=zscore>=exitZ
    short_entry=zscore>entryZ
    short_exit=zscore<=exitZ

    nunitslong = np.full(yport.shape[0], np.nan)
    nunitsshort = np.full(yport.shape[0], np.nan)

    nunitslong[0] = 0
    nunitslong[long_entry] = 1
    nunitslong[long_exit] = 0
    nunitsshort[0] = 0
    nunitsshort[short_entry] = -1
    nunitsshort[short_exit] = 0
    nunitslong=fillMissingData(nunitslong)
    nunitsshort=fillMissingData(nunitsshort)

    nUnits3=nunitslong+nunitsshort
    pos3=(nUnits3[:,None]*r*y2[lookback-1:])
    pos3_lag=pos3[:-1]
    y2_lag=y2[lookback-1:][:-1]
    y2_curr=y2[lookback-1:][1:]
    pnl=np.sum(pos3_lag*(y2_curr-y2_lag)/y2_lag,axis=1)
    denom = np.sum(np.abs(pos3_lag), axis=1)
    ret=np.zeros_like(pnl)
    mask=denom > 0
    ret[mask]=pnl[mask]/denom[mask]
    cp=(np.cumprod(1+ret)-1)
    plt.clf()
    plt.plot(cp)
    plt.savefig('returns_bollinger.png')

