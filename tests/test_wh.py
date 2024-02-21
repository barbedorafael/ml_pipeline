# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:45:17 2024

@author: RafBar
"""
import pandas as pd
import geopandas as gpd

df = pd.read_parquet('data/processed/data4ml_bho.parquet')

wh = gpd.read_file('data/raw/bho_LB/bho_5k_ANA.shp')

wh = wh.set_index('cotrecho')

wh.index.duplicated().sum()

wh.cotrecho.duplicated().sum()

df.loc[:, 'Wavg'] = wh.Largura_me
df['Havg'] = wh.Prof_media


target = 'Wavg'
df = df.loc[~df[target].isna()]
df[target] = df[target]/df.A

targets = ['qm', 'q95', 'Wavg', 'Havg']
models = ['MLR', 'DT', 'KNN', 'SVM', 'GBM', 'RF']
selected_features = df.columns.drop(targets)

for mlmodel in models:
    print('Running for ' + target + ' and ' + mlmodel)
    
    y, result, imps = model_run(df,
                                    selected_features,
                                    target,
                                    mlmodel,
                                    method='k-fold',
                                    )
    plot_results(result, y, imps, target, mlmodel, savefigs=False)
    
    # Creating the DataFrame with specified columns and errors
    dfe = pd.DataFrame(index=df.index)
    dfe[target + '_obs'] = y
    dfe[target + '_pred'] = result