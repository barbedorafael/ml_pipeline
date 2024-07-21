# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 06:18:57 2023

@author: RafBar
"""

# import os
# os.chdir('ml_pipeline')

import numpy as np
import pandas as pd
from src import functions as mlp


df = pd.read_parquet('data/processed/data4ml_gauges.parquet') # gauges or bho
df = df.loc[:, ~(df==0).all(axis=0)]
df = df.drop(['code', 'g_area', 'g_lat', 'g_lon'], axis=1)

# mlp.plot_correlation_matrix(df)

# Choose target
targets = ['qm', 'q95'] # ['Wavg', 'Havg'] #
models = ['MLR', 'DT', 'KNN', 'SVM', 'GBM', 'RF']
features = df.columns.drop(targets)

df = df.sample(frac = 1) # Shuffle values MAKES ALL THE DIFFERENCE IDKWK
for target in targets:
    if df[target].isna().any():
        method = 'dataset'
        selected_features = features
    else:
        method = 'k-fold'
        # method = 'loo'
        # Select features for modelling based on hyerarchical clustering
        cluster_feature, selected_features = mlp.fs_hcluster(df,
                                                             features, 
                                                             target, 
                                                             cluster_threshold=0.4, 
                                                             link_method='average',
                                                             plot=True)
    X = df[selected_features].values
    y = df[target].values
    # y = np.maximum(y, 0.001) # No need because smallest value is 0.075
    
    for mlmodel in models:
        print('Running for ' + target + ' and ' + mlmodel)
        # y = np.log1p(y)
        yhat, imps = mlp.model_run(X,
                                  y,
                                  mlmodel,
                                  method=method,
                                  )
        
        try:
            imps.columns = selected_features
            imps.to_parquet('data/output/imps_'+target+'_'+mlmodel+'_'+method+'.parquet')
            mlp.plot_results(y, yhat, imps, target, mlmodel, savefigs=False)
        except:
            0
        
        # Creating the DataFrame with specified columns and errors
        dfr = pd.DataFrame(index=df.index)
        dfr['obs'] = y
        dfr['pred'] = yhat

        dfr.to_parquet('data/output/results_raw_'+target+'_'+mlmodel+'_'+method+'.parquet')
        
        
        
    

