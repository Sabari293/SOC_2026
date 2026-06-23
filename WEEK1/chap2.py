import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import VarianceRatio
from hurst import compute_Hc
import statsmodels.api as sm
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



def pnl(y3, result):
    beta=result.evec[:, 0]
    p=y3@beta
    p_lag=p[:-1]
    diff=np.diff(p)
    X=sm.add_constant(p_lag)
    model=sm.OLS(diff, X).fit()
    half_life=-np.log(2)/model.params[1]
    lb=int(half_life)
    ma=moving_average(p, lb)
    std=moving_std(p, lb, ma)
    p2=p[lb-1:]
    numUnits=-(p2-ma)/std
    positions=(numUnits[:, None]*beta[None, :]*y3[lb-1:])
    positions_lag=positions[:-1]
    y3_cut=y3[lb-1:]
    y3_lag=y3_cut[:-1]
    y3_curr=y3_cut[1:]
    pnl1=np.sum(positions_lag*(y3_curr-y3_lag)/y3_lag,axis=1)
    ret=pnl1/np.sum(np.abs(positions_lag), axis=1)
    return ret


if __name__=='__main__':

    T = pd.read_csv('DEXCAUS1.csv')
    dates = pd.to_datetime(T['observation_date'])
    usd_cad = T['DEXCAUS'].dropna()

    result = adfuller(usd_cad, maxlag=1)
    h = int(result[1] < 0.05)

    print('ADF test on USD.CAD')
    print(f'  h = {h}')
    print(f'  p-value = {result[1]:.6f}')
    print(f'  Test statistic = {result[0]:.6f}')
    print('  Critical values:')
    print(result[4])

    log_returns = np.diff(np.log(usd_cad))

    result = adfuller(log_returns, maxlag=1)
    h = int(result[1] < 0.05)

    print('\nADF test on log returns USD.CAD')
    print(f'  h = {h}')
    print(f'  p-value = {result[1]:.6f}')
    print(f'  Test statistic = {result[0]:.6f}')
    print('  Critical values:')
    print(result[4])

    usd_cadl = np.log(usd_cad)

    H, c, data = compute_Hc(usd_cad, kind='price', simplified=True)

    print(f'\nHurst exponent: {H:.6f}')

    vr = VarianceRatio(usd_cadl)

    h = int(vr.pvalue < 0.05)

    print('\nVariance ratio test results:')
    print(f'  h = {h}')
    print(f'  p-value = {vr.pvalue:.6f}')

    usd_cad_lag = usd_cad.shift(1)
    diff1 = usd_cad - usd_cad_lag

    df = pd.concat([diff1, usd_cad_lag], axis=1).dropna()
    df.columns = ['diff1', 'lag']

    X = sm.add_constant(df['lag'])
    model = sm.OLS(df['diff1'], X).fit()

    beta = model.params['lag']
    half_life1 = -np.log(2) / beta

    print(f'\nHalf-life (price series): {half_life1:.6f}')

    usd_cadl_lag = usd_cadl.shift(1)
    diff1 = usd_cadl - usd_cadl_lag

    df = pd.concat([diff1, usd_cadl_lag], axis=1).dropna()
    df.columns = ['diff1', 'lag']

    X = sm.add_constant(df['lag'])
    model = sm.OLS(df['diff1'], X).fit()

    beta = model.params['lag']
    half_life2 = -np.log(2) / beta

    print(f'Half-life (log price series): {half_life2:.6f}')
    if(half_life1>=365):
        raise "Half life too large"
    #example 2.5
    usd_cad=usd_cad.to_numpy()
    lookback=int(half_life1)
    ma=moving_average(usd_cad,lookback)
    std=moving_std(usd_cad,lookback,ma)
    y=usd_cad[lookback-1:]
    mktvalue=-(y-ma)/std
    mkt_lag=mktvalue[:-1]
    ylag=y[:-1]
    y_curr=y[1:]
    pnl1=mkt_lag*(y_curr-ylag)/ylag
    cpnl=np.cumsum(pnl1)
    plt.plot(cpnl)
    plt.savefig('pnl1.png')
    plt.clf()

    data = loadmat('inputData_ETF.mat')

    tday = data['tday']
    syms = data['syms']
    cl = data['cl']

    syms = [s[0] for s in syms.squeeze()]

    idxA = np.where(np.array(syms) == 'EWA')[0][0]
    idxB = np.where(np.array(syms) == 'IGE')[0][0]
    idxC = np.where(np.array(syms) == 'EWC')[0][0]
    yA=cl[:,idxA][:,None]
    yB=cl[:,idxC]
    yC=cl[:,idxB]

    dates=pd.to_datetime(tday.flatten().astype(str), format='%Y%m%d')

    plt.scatter(yA, yB, facecolors='none', edgecolors='blue')
    plt.xlabel('EWA')
    plt.ylabel('EWC')
    plt.savefig('ewa_ewc.png')
    plt.clf()

    x=np.hstack((yA,np.ones_like(yA)))
    w=np.linalg.pinv(x) @ yB
    print("Hedge ratio =", w[0])
    print("Intercept =", w[1])

    yB_hat=w[0]*yA[:,0]+w[1]
    plt.plot(dates,yB, label='EWC')
    plt.plot(dates,yB_hat, label='Fitted EWC')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('ewa_ewc_fit.png')
    plt.clf()

    spread=yB-yB_hat
    plt.plot(dates, spread)
    plt.xlabel('Date')
    plt.ylabel('Spread')
    plt.title('Spread = EWC - beta*EWA - intercept')
    plt.savefig('spread.png')
    plt.clf()

    result=engle_granger(yB, yA,trend='c')
    print(result.stat)
    print(result.pvalue)
    print(result.critical_values)

    result=coint_johansen(np.column_stack((yA,yB)),0,1)
    print("Trace statistics:")
    print(result.lr1)
    print("Critical values:")
    print(result.cvt)
    print("Eigen statistics:")
    print(result.lr2)
    print("Critical values :")
    print(result.cvm)
    # print(result.evec)
    plt.plot(result.evec[0,0]*yA[:,0]+yB*(result.evec[1,0]))
    plt.savefig("EWA_EWC_spread.png")
    plt.clf()


    result=coint_johansen(np.column_stack((yA,yB,yC)),0,1)
    print("Trace statistics:")
    print(result.lr1)
    print("Critical values:")
    print(result.cvt)
    print("Eigen statistics:")
    print(result.lr2)
    print("Critical values :")
    print(result.cvm)
    print(result.evec)
    plt.plot(result.evec[0,0]*yA[:,0]+yB*(result.evec[1,0])+yC*(result.evec[2,0]))
    plt.savefig("EWA_EWC_IGE_spread.png")
    plt.clf()



    yA=yA[:,0]
    y3=np.column_stack((yA,yB,yC))


    # print(half_life3)

    pnl2=pnl(y3,result)
    cp=np.cumprod(pnl2+1)-1
    plt.plot(cp)
    plt.savefig('cumulative_pnl_EWA_EWC_IGE.png')
    # print(y3)
