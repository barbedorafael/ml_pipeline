# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:45:17 2024

@author: RafBar
"""

import pandas as pd
import geopandas as gpd

df = pd.read_parquet('data/processed/data4ml_bho.parquet')
df = df.drop(['qm', 'q95'], axis=1)

results_qm = pd.read_parquet('data/output/results_raw_qm_RF_dataset.parquet')
results_q95 = pd.read_parquet('data/output/results_raw_q95_RF_dataset.parquet')

df.loc[:, 'qm_RF'] = results_qm.pred
df.loc[:, 'q95_RF'] = results_q95.pred

wh = gpd.read_file('data/raw/bho_LB/bho_5k_ANA.shp')
wh = wh.set_index('cotrecho')

# df.loc[:, 'Qm'] = wh.Vazao_medi
df.loc[:, 'Wavg'] = wh.Largura_me
df['Havg'] = wh.Prof_media

df.loc[:, 'Wavg'] /= df.loc[:, 'A']
df.loc[:, 'Havg'] /= df.loc[:, 'A']

df.loc[:, 'Wavg'] *= 1e3
df.loc[:, 'Havg'] *= 1e3

results_qm = pd.read_parquet('data/output/results_raw_qm_RF_dataset.parquet')
results_q95 = pd.read_parquet('data/output/results_raw_q95_RF_dataset.parquet')

df.to_parquet('data/processed/data4ml_wh_bho.parquet')

df_gauges = df[df.Havg>0]

df_gauges.to_parquet('data/processed/data4ml_wh_gauges.parquet')
