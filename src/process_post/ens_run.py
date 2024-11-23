# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:55:13 2024

@author: rafbar
"""

import numpy as np
import pandas as pd
from src import functions as mlp

# Choose targets
targets = ['qm', 'q95']
models =  ['SVM', 'GBM', 'RF']
methods = ['k-fold', 'dataset']

df = pd.read_parquet('data/processed/data4ml_gauges.parquet')
df = df.loc[:, ~(df==0).all(axis=0)]
# df = df.drop(['code', 'g_area', 'g_lat', 'g_lon'], axis=1)
# df = df.drop(['cocursodag', 'cobacia', 'nucomptrec'], axis=1)

for method in methods:
    for target in targets:
        ens = []
        ens_imps = []
        for mlmodel in models:
            
            dfr = pd.read_parquet('data/output/results_raw_{}_{}_{}.parquet'.format(target, mlmodel, method))
            ens.append(dfr.pred)
            
            if method=='k-fold':
                imps = pd.read_parquet('data/output/imps_{}_{}_{}.parquet'.format(target, mlmodel, method))
                ens_imps.append(imps)
            
        ens_mean = np.array(ens).mean(axis=0)
        dfr.pred = ens_mean
        
        dfr.to_parquet('data/post/results_raw_{}_ensemble_{}.parquet'.format(target, method))
        
        #### Post processing
        dfr['pred'] = np.maximum(dfr.pred, 0.001)
        
        # Ensemble std and cv
        ens = np.maximum(ens, 0.001)
        ens_std = np.array(ens).std(axis=0)
        ens_cv = ens_std / dfr.pred.values
        dfr['ens_std'] = ens_std
        dfr['ens_cv'] = ens_cv
        
        if method=='k-fold':
            # mlp.plot_results(dfr.obs.values, dfr.pred.values, imps, target, mlmodel, savefigs=False)
            
            # Errors
            dfr['error_abs'] = dfr.obs - dfr.pred
            dfr['error_rel'] = dfr.error_abs / dfr.pred
            dfr[['lat', 'lon']] = df[['g_lat', 'g_lon']]
            
            df_imps = pd.concat(ens_imps)
            df_imps.to_parquet('data/post/imps_{}_ensemble_{}.parquet'.format(target, method))

        # post-processed results
        dfr.to_parquet(f'data/post/results_post_{target}_ensemble_{method}.parquet')
        dfr.to_csv(f'data/post/results_post_{target}_ensemble_{method}.csv', index=False)













