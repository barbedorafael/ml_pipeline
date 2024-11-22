# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 06:18:57 2023

@author: RafBar
"""

# import os
# os.chdir('ml_pipeline')

import numpy as np
import pandas as pd
import functions as mlp


df = pd.read_parquet('data/processed/data4ml_gauges.parquet') # gauges or bho
df = df.loc[:, ~(df==0).all(axis=0)]
df = df.drop(['code', 'g_area', 'g_lat', 'g_lon'], axis=1)

# mlp.plot_correlation_matrix(df)

# Choose target
targets = ['qm', 'q95']
models = ['MLR', 'DT', 'KNN', 'SVM', 'GBM', 'RF']
features = df.columns.drop(targets)

df = df.sample(frac = 1) # Shuffle values
for target in targets:
    if df[target].isna().any():
        method = 'dataset'
        selected_features = features
    else:
        method = 'k-fold'
        # Select features for modelling based on hyerarchical clustering
        cluster_feature, selected_features = mlp.fs_hcluster(df,
                                                             features, 
                                                             target, 
                                                             cluster_threshold=0.4, 
                                                             link_method='average',
                                                             plot=True)

    X = df[selected_features].values
    y = df[target].values
    
    for mlmodel in models:
        print('Running for ' + target + ' and ' + mlmodel)
        yhat, imps = mlp.model_run(X,
                                  y,
                                  mlmodel,
                                  method=method,
                                  )
        
        try:
            imps.columns = selected_features
            mlp.plot_results(y, yhat, imps, target, mlmodel, savefigs=False)
            
            imps.to_parquet('data/output/imps_'+target+'_'+mlmodel+'_'+method+'.parquet')
        except:
            0
        
        # Creating the DataFrame with specified columns and errors
        dfr = pd.DataFrame(index=df.index)
        dfr['obs'] = y
        dfr['pred'] = yhat

        dfr.to_parquet('data/output/results_raw_'+target+'_'+mlmodel+'_'+method+'.parquet')
        
        
        
    

