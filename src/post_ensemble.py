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
models = ['SVM', 'GBM', 'RF']
methods = ['k-fold', 'dataset']

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
        ens_std = np.array(ens).std(axis=0)
        ens_cv = ens_std / ens_mean
        dfr.pred = ens_mean
        dfr['std'] = ens_std
        dfr['cv'] = ens_cv
        
        dfr.to_parquet('data/post/results_raw_{}_ensemble_{}.parquet'.format(target, method))
        
        if method=='k-fold':
            mlp.plot_results(dfr.obs.values, dfr.pred.values, imps, target, mlmodel, savefigs=True)
            
            df_imps = pd.concat(ens_imps)
            df_imps.to_parquet('data/post/imps_{}_ensemble_{}.parquet'.format(target, method))
