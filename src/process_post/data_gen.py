# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 15:32:04 2024

@author: rafbar
"""

import numpy as np
import pandas as pd
import geopandas as gpd

base = pd.read_parquet('data/external/bho5k_trecho.parquet').set_index('cotrecho', drop=True).sort_index()
base = base.dropna(subset='cocursodag')
base = base[~(base.dsversao.str.startswith('BHO ilha'))]
base = base[(base.cobacia.str.startswith('4')) | 
            ~(base.dedominial.isin(['Internacional', 'Linha de Costa']))]
base = base[['cobacia', 'nuareamont']]
base = base.rename(columns={'cobacia': 'code_otto', 'nuareamont': 'area_upst'})
base.index.name = 'seg_id'

# Choose targets
targets = ['qm', 'q95']
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
    bho_r = pd.read_parquet('data/post/results_post_'+target+'_'+mlmodel+'_dataset.parquet')
    # bho_r.loc[dfr.index, 'pred'] = dfr.pred # Replace training predictions with k-fold predictions
    
    dataset = base.copy()
    dataset['pred'] = bho_r.pred
    dataset['ens_std'] = bho_r.ens_std
    dataset['ens_cv'] = bho_r.ens_cv
    
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


### Process environmental descriptors dataset
bho_d = pd.read_parquet('data/processed/data4ml_bho.parquet')
bho_d = bho_d.drop(['code', 'g_area', 'g_lat', 'g_lon', 'qm', 'q95'], axis=1)

dataset = base.join(bho_d)

for geom_type in geom_types:
    bho_geom = gpd.read_parquet('data/external/bho5k_{}.parquet'.format(geom_type),
                                columns=['cotrecho', 'geometry']).set_index('cotrecho', drop=False).sort_index()
    gdf = gpd.GeoDataFrame(dataset, geometry=bho_geom.geometry)
    gdf.to_parquet('data/post/bho_{}_descriptors.parquet'.format(geom_type))

