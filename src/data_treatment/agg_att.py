# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:02:38 2024

@author: rafbar
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import ast

subs = gpd.read_parquet('data/raw/bho_flow_data.parquet')
subs = subs.sort_values(by='cotrecho', ignore_index=True)

centroids = subs.geometry.representative_point()
lats = centroids.y.rename('lat')
lons = centroids.x.rename('lon')

def open_attribute_data(file):
    df = pd.read_csv(file, index_col='system:index')
    df = df.sort_values(by='cotrecho', ignore_index=True)
    df = df.drop(['.geo'], axis=1)
    return df

att_cli = open_attribute_data('data/raw/bho_attributes_climate.csv')
att_soil = open_attribute_data('data/raw/bho_attributes_soilcontent.csv')
att_texlit = open_attribute_data('data/raw/bho_attributes_textlito.csv')
att_topolc = open_attribute_data('data/raw/bho_attributes_topolc.csv')
att_ws = open_attribute_data('data/raw/bho_attributes_ws.csv')

att = pd.concat([att_cli, att_soil, att_texlit, att_topolc, att_ws], axis=1)
att = att.loc[:,~att.columns.duplicated(keep='last')]

def unravel_classes(attribute, classes):
    df = pd.DataFrame(att[attribute].str.replace('=',':').apply(
            lambda x: ast.literal_eval(x)).to_list(), columns=np.arange(1,len(classes)+1))
    df.columns = classes
    psum = df.sum(axis=1)
    return (df.div(psum, axis=0) * 100).round(2)

dd = att['sum'].div(att['nuareacont']*1000, axis=0).rename('dd').fillna(0)
dd = dd/9 # Correct from EE collection
tc_df = unravel_classes('terrain', 
                        ['tc_wetland', 'tc_flat', 'tc_gentle', 'tc_moderate', 'tc_steep', 'tc_extreme']).fillna(0)
lc_df = unravel_classes('landcover', 
                        ['lc_forest', 'lc_grassland', 'lc_agriculture', 'lc_permeable', 'lc_water']).fillna(0)
st_df = unravel_classes('soiltext', 
                        ['st_Cl', 'st_SiCl', 'st_SaCl', 'st_ClLo', 'st_SiClLo', 'st_SaClLo', 'st_Lo', 
                         'st_SiLo', 'st_SaLo', 'st_Si', 'st_LoSa', 'st_Sa']).fillna(0)
lt_df = unravel_classes('lithology', 
                        ['lt_Su', 'lt_Vb', 'lt_Ss', 'lt_Pb', 'lt_Sm', 'lt_Sc', 'lt_Va', 'lt_Mt', 'lt_Pa', 
                         'lt_Vi', 'lt_Wb', 'lt_Py', 'lt_Pi', 'lt_Ev', 'lt_Nd', 'lt_Ig']).fillna(0)

data = pd.concat([att.drop(['sum', 'landcover', 'terrain', 'soiltext', 'lithology'], axis=1), 
                  dd, tc_df, lc_df, st_df, lt_df,
                  lats, lons, 
                  subs.drop(['nustrahler', 'dedominial', 'geometry'], axis=1)
                  ], axis=1)

data = data.loc[:,~data.columns.duplicated(keep='last')]

# Check NANs
nancount = data.isna().sum()

# Change dtype of codified attributes
data = data.dropna(subset=['cocursodag'])
data.loc[:, ['cobacia', 'cocursodag']] = data.loc[:, ['cobacia', 'cocursodag']].astype(str)

data.to_parquet('data/bho_data_unit.parquet')