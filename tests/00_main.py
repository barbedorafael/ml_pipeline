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


df = pd.read_parquet('data/processed/data4ml_gauges.parquet')
df = df.loc[:, ~(df==0).all(axis=0)]
df = df.drop(['code', 'g_area', 'g_lat', 'g_lon'], axis=1)
df = df.drop(['cocursodag', 'cobacia', 'nucomptrec'], axis=1)

# Choose target (qm or q95)
# target = 'q95' # q95 or qm
targets = ['qm', 'q95'] #, 'Wavg', 'Havg']
models = ['MLR', 'DT', 'KNN', 'SVM', 'GBM', 'RF']
features = df.columns.drop(targets)

df = df.sample(frac = 1) # Shuffle values MAKES ALL THE DIFFERENCE IDKW
for target in targets:
    # error = pd.read_parquet('results_'+target+'_'+mlmodel+'.parquet')
    
    try:
        has_data = df.has_data
        method = 'dataset'
        selected_features = features.drop('has_data')
    except:
        has_data = None
        method = 'k-fold'
        # Select features for modelling based on hyerarchical clustering
        cluster_feature, selected_features = mlp.fs_hcluster(df,
                                                             features, 
                                                             target, 
                                                             cluster_threshold=0.4, 
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
                                  has_data=has_data
                                  )
        imps.columns = selected_features
        
        # y = np.expm1(y)
        # yhat = np.expm1(yhat)
        
        # Creating the DataFrame with specified columns and errors
        dfr = pd.DataFrame(index=df.index)
        dfr['obs'] = y
        dfr['pred'] = yhat

        imps.to_parquet('data/output/imps_'+target+'_'+mlmodel+'_'+method+'.parquet')
        dfr.to_parquet('data/output/results_raw_'+target+'_'+mlmodel+'_'+method+'.parquet')
        
        mlp.plot_results(y, yhat, imps, target, mlmodel, savefigs=True)
        
    

