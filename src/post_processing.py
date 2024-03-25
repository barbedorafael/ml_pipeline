# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:08:29 2024

@author: RafBar
"""

import numpy as np
import pandas as pd
from src import functions as mlp
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


df = pd.read_parquet('data/processed/data4ml_gauges.parquet')
df = df.loc[:, ~(df==0).all(axis=0)]
# df = df.drop(['code', 'g_area', 'g_lat', 'g_lon'], axis=1)
# df = df.drop(['cocursodag', 'cobacia', 'nucomptrec'], axis=1)

# Choose target (qm or q95)
# target = 'q95' # q95 or qm
targets = ['qm', 'q95'] #, 'Wavg', 'Havg']
models = ['SVM', 'GBM', 'RF']
features = df.columns.drop(targets)

for target in targets:
    for mlmodel in models:
        
        dfr = pd.read_parquet('data/output/results_raw_'+target+'_'+mlmodel+'_k-fold.parquet')
        imps = pd.read_parquet('data/output/imps_'+target+'_'+mlmodel+'_k-fold.parquet')
        
        dfr = dfr.sort_index()
        # min_value = dfr.obs.quantile([0.01]).iloc[0]
        dfr['pred'] = np.maximum(dfr.pred, 0.01)
        
        # dfr = dfr[dfr.pred < 12.5]
        
        y = dfr['obs'].values
        yhat = dfr['pred'].values
        

        # mlp.plot_results(y, yhat, imps, target, mlmodel, savefigs=False)
        
        dfr['error'] = y - yhat
        
        dfcorrx = df.corrwith(abs(dfr.error), method='spearman')
        dfcorry = dfr.corrwith(abs(dfr.error), method='spearman')
        # print(dfcorry)
        
        step = 0.1
        steps = np.arange(step, 1.05, step).round(2)
        quantiles = dfr.pred.quantile(steps)
        # quantiles = dfr.pred.quantile([0.05, 0.25, 0.5, 0.75, 0.95, 1])
        lower = 0
        e_range = pd.DataFrame(columns=['90%_low', '90%_high', '75%_low', '75%_high'])
        for q, upper in quantiles.items():
            subset = (dfr.pred > lower) & (dfr.pred <= upper)
            dfq = dfr[subset]
            e_range.loc[q, '90%_low'] = dfq.error.quantile([0.05]).iloc[0]
            e_range.loc[q, '90%_high'] = dfq.error.quantile([0.95]).iloc[0]
            e_range.loc[q, '75%_low'] = dfq.error.quantile([0.125]).iloc[0]
            e_range.loc[q, '75%_high'] = dfq.error.quantile([0.875]).iloc[0]
            lower = upper
            
        e_range['upper_limit'] = quantiles
        xdata = e_range.upper_limit.rolling(2).mean()
        xdata.iloc[0] = e_range.upper_limit.iloc[0] / 2
        xdata = xdata.values
        
        dfq = dfr.copy().sort_values(by='pred')

        # Quantile models
        params = pd.DataFrame(columns=['90%_low', '90%_high', '75%_low', '75%_high'])
        
        for n, qd in enumerate([0.05, 0.95, 0.125, 0.875]):
            quantile_model = smf.quantreg('error ~ pred', dfq).fit(q=qd)
            params.iloc[:,n] = quantile_model.params
        
        xx = np.arange(0.1, dfq.pred.iloc[-1], 0.1)
        plt.scatter(dfr['pred'], dfr['error'], s=15, alpha=0.1, label='Data Points')
        for n, column in enumerate(params.columns):
            a = params.loc['pred', column]
            b = params.loc['Intercept', column]
            
            yy = a * xx + b
            yy[(xx + yy) < 0] = -xx[(xx + yy) < 0]
            plt.plot(xx, yy, label='Quantile Fit ' + str(column))
            
            ydata_noisy = e_range.iloc[:, n].values
            plt.scatter(xdata, ydata_noisy)
        
        plt.xlim([-0.5, dfq.pred.quantile(0.95)])
        plt.ylim([-5, 5])
        # plt.ylim([-1, 2.5])
        # plt.ylim([dfr.error.quantile(0.1), dfr.error.max()])
        plt.title(mlmodel + ' ' + target)
        # plt.legend()
        plt.show()
        
        
        dataset = pd.read_parquet('data/output/results_raw_'+target+'_'+mlmodel+'_dataset.parquet')
        dataset.loc[dfr.index, 'pred'] = dfr.pred # Replace training predictions with k-fold predictions
        dataset = dataset.drop('obs', axis=1)
        
        # Iterate through each column in the original DataFrame
        for column in params.columns:
            a = params.loc['pred', column]
            b = params.loc['Intercept', column]
            
            # Apply the calculation a*x + b
            dataset[column] = dataset.pred * (1 + a) + b
        
        dataset[dataset<0] = 0
        dataset['gauged'] = dataset.index.isin(dfr.index)
        
        # dataset.to_parquet('data/output/bho_'+target+'_'+mlmodel+'.parquet')





