# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 06:18:57 2023

@author: RafBar
"""

import pandas as pd
import ml_pipeline as mlp

df = pd.read_parquet('data/data4ml.parquet')
df = df.sample(frac = 1) # Shuffle values MAKES ALL THE DIFFERENCE IDKW
df = df.loc[:, ~(df==0).all(axis=0)]
df = df.drop(['code', 'g_area', 'g_lat', 'g_lon'], axis=1)

# Choose target (qm or q95)
# target = 'q95' # q95 or qm
targets = ['qm', 'q95']
models = ['MLR', 'DT', 'KNN', 'SVM', 'GBM', 'RF']

for target in targets:

    try:
        df.has_data
        method = 'dataset'
        selected_features = df.columns[:-3]
    except:
        method = 'k-fold'
        # Select features for modelling based on hyerarchical clustering
        cluster_feature, selected_features = mlp.feature_selection(df, target, cluster_threshold=0.4, plot=False)
    
    
    for mlmodel in models:
        print('Running for ' + target + ' and ' + mlmodel)
        error = pd.read_parquet('results_'+target+'_'+mlmodel+'.parquet')
        
        y, result, imps = mlp.model_run(df,
                                        selected_features,
                                        target,
                                        mlmodel,
                                        method=method,
                                        )
        # mlp.plot_results(result, y, imps, target, mlmodel, savefigs=True)
        
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
        
    

