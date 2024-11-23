# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 20:41:45 2024

@author: rafbar
"""

import pandas as pd
import geopandas as gpd
import ast

data_dir = 'data/external/'

trecs = gpd.read_parquet(data_dir + 'bho5k_trecho.parquet',
        columns = ['cotrecho', 'cobacia', 'cocursodag', 'nutrjus', 
                          'nuareacont', 'nuareamont', 'nucomptrec', 'nustrahler',
                          'dedominial', 'geometry'])

qdata = gpd.read_file(data_dir + 'base_bho_baciasinc_qinc_sem_climadj.gpkg',
                      include_fields=['cotrecho', 'posto', 'p1_qmca', 'p1_q95ca'])

gaugelocs = gpd.read_file(data_dir + 'base_postos_ana.gpkg',
            include_fields=['codigo', 'area_km2', 'latitude', 'longitude']) # THIS INCLUDES MORE GAUGES THAN THE ONES USED

subs = pd.merge(qdata.drop('geometry', axis=1), gaugelocs.drop('geometry', axis=1),
                how='left', left_on='posto', right_on='codigo').drop('posto', axis=1) # THIS INCLUDES ALL BHO BUT DATA ONLY AT GAUGED SITES

subs = subs.rename({'codigo': 'code',
                    'p1_qmca': 'qm',
                    'p1_q95ca': 'q95',
                    'area_km2': 'g_area',
                    'latitude': 'g_lat',
                    'longitude': 'g_lon'},
                    axis=1)

df_work = pd.merge(trecs.drop('geometry', axis=1), subs, on='cotrecho')

gdf = gpd.GeoDataFrame(df_work, geometry=trecs.geometry)

gdf.to_parquet('data/raw/bho_flow_data.parquet')

## Only gauges geom
gauges = subs.dropna(subset='code')
gauges = gauges.merge(gaugelocs[['codigo', 'geometry']], left_on='code', right_on='codigo').drop('codigo', axis=1)
gauges = gpd.GeoDataFrame(gauges, crs='epsg:4326')
gauges.to_file('data/raw/gauge_data_points.fgb', driver='FlatGeobuf')
