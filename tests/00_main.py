# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 06:18:57 2023

@author: RafBar
"""

import os
os.chdir('ml_pipeline')

import pandas as pd
from src import functions as mlp


df = pd.read_parquet('data/processed/data4ml_gauges.parquet')
df = df.sample(frac = 1) # Shuffle values MAKES ALL THE DIFFERENCE IDKW
df = df.loc[:, ~(df==0).all(axis=0)]
df = df.drop(['code', 'g_area', 'g_lat', 'g_lon'], axis=1)
df = df.drop(['lat', 'lon', 'cocursodag', 'cobacia', 'nucomptrec', 'L'], axis=1)

# Choose target (qm or q95)
# target = 'q95' # q95 or qm
targets = ['qm', 'q95'] #, 'Wavg', 'Havg']
models = ['MLR', 'DT', 'KNN', 'SVM', 'GBM', 'RF']
features = df.columns.drop(targets)

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
    
    for mlmodel in models:
        print('Running for ' + target + ' and ' + mlmodel)
                
        y, result, imps = mlp.model_run(X,
                                        y,
                                        mlmodel,
                                        method=method,
                                        has_data=has_data
                                        )
        mlp.plot_results(result, y, imps, target, mlmodel, savefigs=False)
        
        # Creating the DataFrame with specified columns and errors
        dfe = pd.DataFrame(index=df.index)
        dfe[target + '_obs'] = y
        dfe[target + '_pred'] = result
        try:
            dfe[target + '_error'] = y - result
        except:
            0
        # imps.to_parquet('data/output/imps_'+target+'_'+mlmodel+'_'+method+'.parquet')
        dfe.to_parquet('data/output/results_'+target+'_'+mlmodel+'_'+method+'.parquet')
        
    

