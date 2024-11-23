# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:04:19 2024

@author: rafbar
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import time

pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_parquet('data/interim/bho_data_unit.parquet')#, engine='pyarrow', dtype_backend='pyarrow')
df = df.sort_values(by='cobacia', ascending=False) # sort by otto code to speed up computations
df = df.set_index('cotrecho')
n = len(df)

values_wavg = ['elv_avg', 'slp_avg', 'hnd_avg',
               '01_P', '02_P', '03_P', '04_P', '05_P', '06_P', 
               '07_P', '08_P', '09_P', '10_P', '11_P', '12_P', 
               '01_T', '02_T', '03_T', '04_T', '05_T', '06_T', 
               '07_T', '08_T', '09_T', '10_T', '11_T', '12_T', 
               '01_ET', '02_ET', '03_ET', '04_ET', '05_ET', '06_ET', 
               '07_ET', '08_ET', '09_ET', '10_ET', '11_ET', '12_ET', 
               '01_PET', '02_PET', '03_PET', '04_PET', '05_PET', '06_PET', 
               '07_PET', '08_PET', '09_PET', '10_PET', '11_PET', '12_PET',
               'TWS', 'GW', 'SM', 'SW', 'RS',
               'dd', 'tc_wetland', 'tc_flat', 'tc_gentle', 'tc_moderate', 'tc_steep', 'tc_extreme',
               'lc_forest', 'lc_grassland', 'lc_agriculture', 'lc_permeable', 'lc_water',
               'soilorga', 'soilclay', 'soilsand', 'soilwate',
               'st_Cl', 'st_SiCl', 'st_SaCl', 'st_ClLo', 'st_SiClLo', 'st_SaClLo', 'st_Lo', 'st_SiLo', 'st_SaLo', 'st_Si', 'st_LoSa', 'st_Sa', # soil texture classes
               'lt_Su', 'lt_Vb', 'lt_Ss', 'lt_Pb', 'lt_Sm', 'lt_Sc', 'lt_Va', 'lt_Mt', 'lt_Pa', 'lt_Vi', 'lt_Wb', 'lt_Py', 'lt_Pi', 'lt_Ev', 'lt_Nd', 'lt_Ig', # lithology classes
               'lat', 'lon']
values_std = ['elv_std', 'slp_std', 'hnd_std']
values_avg = ['elv_avg', 'slp_avg', 'hnd_avg']
# dwnst = True

#### FOR OPERATION (every cobasin)

j=0
stateprint = 0
# ups = df[df['nustrahler']==1]
df_proof = df.copy()
# values2append = pd.DataFrame(columns=df.columns)
# df['flag'] = 0
df['L'] = df['nucomptrec']
df['A'] = df['nuareacont']

start_time = time.time()

for k, kdown in df.query('nutrjus in cotrecho').nutrjus.items():
    
    j+=1
    state = j/n*100
    if state - stateprint > 0.1:
        stateprint = state
        print('{:.5} %'.format(stateprint))
        
        end_time = time.time()
        print('Time: ' + str(((end_time - start_time)/60)) + ' min')
    
    # Find downstream stretch
    trecs_in = df[df.index.isin([k, kdown])]
    
    wavg = trecs_in[values_wavg]#.to_numpy()
    w = trecs_in['A']#.to_numpy()
    
    # if wavg.isna().any().any():
    #     break
    
    att_avg = wavg.multiply(w, axis=0).sum().divide(w.sum())
    
    # wavg = np.ma.masked_array(wavg, np.isnan(wavg))
    # att_avg = np.ma.average(wavg, axis=0, weights=w).filled(np.nan) # 0.005s
    # att_std = w_std(std, avg, w) # 0.004s
    
    # kdown = kdown.values[0]
    df.loc[kdown, 'L'] = trecs_in['L'].sum()
    df.loc[kdown, 'A'] = trecs_in['A'].sum() # 0.001
    df.loc[kdown, values_wavg] = att_avg # 0.003☺
    # df.loc[kdown, values_std] = att_std # 0.001    

    # print('New values:')
    # print(df.loc[kdown, ['cobacia', 'nuareacont', 'nuareamont', 'A']])
    # print('\n')

end_time = time.time()
print('Time to process: ' + str(((end_time - start_time)/60)) + ' min')

check_areas = df[['A', 'nuareamont', 'nuareacont', 'cobacia']]
df = df.drop(['A', 'nutrjus', 'nuareacont'], axis=1)

df.to_parquet('data/interim/bho_data_agg.parquet')
