# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:08:29 2024

@author: RafBar
"""

import numpy as np
import pandas as pd
from src import functions as mlp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


df = pd.read_parquet('data/processed/data4ml_gauges.parquet')
df = df.loc[:, ~(df==0).all(axis=0)]
# df = df.drop(['code', 'g_area', 'g_lat', 'g_lon'], axis=1)
# df = df.drop(['cocursodag', 'cobacia', 'nucomptrec'], axis=1)

# Choose target (qm or q95)
# target = 'q95' # q95 or qm
targets = ['qm', 'q95'] #, 'Wavg', 'Havg']
models = ['MLR', 'DT', 'KNN', 'SVM', 'GBM', 'RF']
features = df.columns.drop(targets)

method = 'k-fold'
for target in targets:
    for mlmodel in models:
        
        dfr = pd.read_parquet('data/output/results_'+target+'_'+mlmodel+'_'+method+'.parquet')
        imps = pd.read_parquet('data/output/imps_'+target+'_'+mlmodel+'_'+method+'.parquet')
        
        dfr = dfr.sort_index()
        # min_value = dfr.obs.quantile([0.025]).iloc[0]
        # dfr['pred'] = np.maximum(dfr.pred, min_value)
        
        # dfr = dfr[dfr.pred < 12.5]
        
        y = dfr['obs'].values
        yhat = dfr['pred'].values
        # mlp.plot_results(y, yhat, imps, target, mlmodel, savefigs=True)
        
        dfr['error'] = y - yhat
        dfr['rr'] = dfr.error / dfr.pred
        
        dfcorrx = df.corrwith(abs(dfr.rr), method='spearman')
        dfcorry = dfr.corrwith(abs(dfr.rr), method='spearman')
        # print(dfcorry)
        
        step = 0.1
        steps = np.arange(step, 1.05, step).round(2)
        quantiles = dfr.pred.quantile(steps)
        # quantiles = dfr.pred.quantile([0.05, 0.25, 0.5, 0.75, 0.95, 1])
        lower = 0
        rr_range = pd.DataFrame(columns=['90%_low', '90%_high', '75%_low', '75%_high'])
        dfu = df[['cobacia', 'cocursodag']]
        dfu.loc[:, 'pred'] = yhat
        dfu.loc[:, rr_range.columns] = 0.0
        for q, upper in quantiles.items():
            subset = (dfr.pred > lower) & (dfr.pred <= upper)
            dfq = dfr[subset]
            rr_range.loc[q, '90%_low'] = dfq.rr.quantile([0.05]).iloc[0]
            rr_range.loc[q, '90%_high'] = dfq.rr.quantile([0.95]).iloc[0]
            rr_range.loc[q, '75%_low'] = dfq.rr.quantile([0.125]).iloc[0]
            rr_range.loc[q, '75%_high'] = dfq.rr.quantile([0.875]).iloc[0]
            lower = upper
            
            # for i, s in rr_range.iterrows():
            #     dfu.loc[subset, s.index.to_list()] = np.outer(dfu.loc[subset, 'pred'], (1+rr_range.loc[q]))
        rr_range['upper_limit'] = quantiles
        
        xdata = rr_range.upper_limit.rolling(2).mean()
        xdata.iloc[0] = rr_range.upper_limit.iloc[0] / 2
        xplot = np.arange(0.01, xdata.max() * 2,0.1)
        
        plt.scatter(dfr['pred'], dfr['rr'], s=3, alpha=0.25, label='Data Points')
        for qd in rr_range.columns[:-1]:
            ydata_noisy = rr_range[qd].values
            popt, pcov = curve_fit(rational_func, xdata, ydata_noisy)
            yplot = rational_func(xplot, *popt)
            # Plot the original data and the fitted curve
            plt.scatter(xdata, ydata_noisy)
            plt.plot(xplot, yplot, 'r-', label=qd)
            
            print(qd)
            # print(np.quantile(yplot, [0.1, 0.25, 0.5, 0.75, 0.9]))
            print(rational_func(xdata, *popt))
            print('\n')
        plt.ylim([-1, 2.5])
        plt.legend()
        plt.show()
        

        


def rational_func(x, a, b, c, d):
    return a + b / x ** 0.5


from scipy.stats import normaltest

stat, p_value = normaltest()
print('Statistics=%.3f, p=%.3f' % (stat, p_value))

# Interpret
alpha = 0.05
if p_value > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


import statsmodels.formula.api as smf

dfq = dfr.copy()
# dfq = dfq[dfq.pred <= dfq.pred.quantile(0.95)]

def transform_x(x):
    threshold = np.quantile(x, 0.95)
    constant_value = 1 / np.sqrt(threshold)  # Constant after the threshold
    transformed_x = np.where(x <= threshold, 1 / np.sqrt(x), constant_value)
    return transformed_x

# dfq['inv_pred'] = 1 / dfq.pred ** 0.5
dfq['inv_pred'] = transform_x(dfq.pred)
dfq = dfq.sort_values(by='pred')

dfq['rr_abs'] = abs(dfq.rr)

plt.scatter(dfr['pred'], dfr['rr'], s=15, alpha=0.1, label='Data Points')
for qd in [0.05, 0.95, 0.125, 0.875]:
    quantile_model = smf.quantreg('rr ~ inv_pred', dfq).fit(q=qd)
    dfq[qd] = quantile_model.predict(dfq)
    plt.plot(dfq.pred, dfq[qd], label='Quantile Fit ' + str(qd))
    print( ((dfq[qd] - dfq.rr) >= 0).mean() )
# plt.xlim([0, xplot.max()])
plt.ylim([-1, 2.5])
plt.legend()
plt.show()
