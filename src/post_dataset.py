# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 15:32:04 2024

@author: rafbar
"""

import numpy as np
import pandas as pd
import geopandas as gpd

dataset = pd.read_parquet('data/external/bho5k_trecho.parquet').set_index('cotrecho', drop=True).sort_index()
dataset = dataset.dropna(subset='cocursodag')
dataset = dataset[~(dataset.dsversao.str.startswith('BHO ilha'))]
dataset = dataset[(dataset.cobacia.str.startswith('4')) | 
                  ~(dataset.dedominial.isin(['Internacional', 'Linha de Costa']))]
dataset = dataset[['cobacia', 'nuareamont']]
dataset = dataset.rename(columns={'cobacia': 'code_otto', 'nuareamont': 'area_upst'})
dataset.index.name = 'seg_id'

# Choose targets
targets = ['qm', 'q95'] # ['Wavg', 'Havg'] # 
mlmodel = 'ensemble'
geom_types = ['trecho', 'area']

for target in targets:
    
    # post-processed results
    dfr = pd.read_parquet('data/post/results_post_'+target+'_'+mlmodel+'_k-fold.parquet')
    
    # empiric uncertainty bands
    e_range = pd.read_csv('data/post/empiric_uncertainty_'+target+'_'+mlmodel+'.csv')
    
    # parameters of quantile regression
    params = pd.read_csv('data/post/parameters_uncertainty_'+target+'_'+mlmodel+'.csv')
    
    
    ### Process dataset
    bho_r = pd.read_parquet('data/post/results_raw_'+target+'_'+mlmodel+'_dataset.parquet')
    # bho_r.loc[dfr.index, 'pred'] = dfr.pred # Replace training predictions with k-fold predictions
    
    dataset['pred'] = bho_r.pred
    dataset['pred'] = np.maximum(dataset.pred, 0)
    
    # dataset['pred'] = dataset.pred * dataset.nuareamont / 1000
    
    # Iterate through each column in the original DataFrame
    for column in params.columns:
        a = params.iloc[-1][column]
        b = params.iloc[0][column]
        
        band_values = dataset.pred * (1 + a) + b
        band_values[band_values<0] = 0
        
        # Apply the calculation a*x + b
        dataset.loc[:, column] = band_values
    
    dataset.loc[:,'gauged'] = dataset.index.isin(dfr.index)
    dataset.to_parquet('data/post/bho_data_{}_{}.parquet'.format(target, mlmodel))
    
    for geom_type in geom_types:
        bho_geom = gpd.read_parquet('data/external/bho5k_{}.parquet'.format(geom_type),
                                    columns=['cotrecho', 'geometry']).set_index('cotrecho', drop=False).sort_index()
        gdf = gpd.GeoDataFrame(dataset, geometry=bho_geom.geometry)
        gdf.to_parquet('data/post/bho_{}_{}_{}.parquet'.format(geom_type, target, mlmodel))





