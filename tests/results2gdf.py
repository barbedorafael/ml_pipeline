# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 20:43:17 2024

@author: RafBar
"""

import pandas as pd
import geopandas as gpd

target = 'q95'
mlmodel = 'RF'
method = 'dataset'

gdf = gpd.read_parquet('data/input/bho_flow_data.parquet')
gdf = gdf.set_index('cobacia')


targets = ['qm', 'q95']
models = ['MLR', 'DT', 'KNN', 'SVM', 'GBM', 'RF']
for target in targets:
    for mlmodel in models:
        qres = pd.read_parquet('data/output/results_'+target+'_'+mlmodel+'_'+method+'.parquet')
        gdf2save = qres[[target+'_obs', target+'_pred']].join(gdf, how='left', on='cobacia')

        gdf2save = gpd.GeoDataFrame(gdf2save)
        gdf2save.to_parquet('data/bhofinal/bho_'+target+'_'+mlmodel+'.parquet')

