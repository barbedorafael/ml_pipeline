# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:54:40 2023

@author: RafBar
"""

import pandas as pd

# values_wavg = ['elv_avg', 'slp_avg', 'hnd_avg',
#                '01_P', '02_P', '03_P', '04_P', '05_P', '06_P', 
#                '07_P', '08_P', '09_P', '10_P', '11_P', '12_P', 
#                '01_T', '02_T', '03_T', '04_T', '05_T', '06_T', 
#                '07_T', '08_T', '09_T', '10_T', '11_T', '12_T', 
#                '01_ET', '02_ET', '03_ET', '04_ET', '05_ET', '06_ET', 
#                '07_ET', '08_ET', '09_ET', '10_ET', '11_ET', '12_ET', 
#                '01_PET', '02_PET', '03_PET', '04_PET', '05_PET', '06_PET', 
#                '07_PET', '08_PET', '09_PET', '10_PET', '11_PET', '12_PET',
#                'TWS', 'GW', 'SM', 'SW', 'RS',
#                'dd', 'tc_wetland', 'tc_flat', 'tc_gentle', 'tc_moderate', 'tc_steep', 'tc_extreme',
#                'lc_forest', 'lc_grassland', 'lc_agriculture', 'lc_permeable', 'lc_water',
#                'soilorga', 'soilclay', 'soilsand', 'soilwate'
#                'st_Cl', 'st_SiCl', 'st_SaCl', 'st_ClLo', 'st_SiClLo', 'st_SaClLo', 'st_Lo', 'st_SiLo', 'st_SaLo', 'st_Si', 'st_LoSa', 'st_Sa', # soil texture classes
#                'lt_Su', 'lt_Vb', 'lt_Ss', 'lt_Pb', 'lt_Sm', 'lt_Sc', 'lt_Va', 'lt_Mt', 'lt_Pa', 'lt_Vi', 'lt_Wb', 'lt_Py', 'lt_Pi', 'lt_Ev', 'lt_Nd', 'lt_Ig', # lithology classes
#                'lat', 'lon']

# df = pd.read_parquet('data/bho_data_agg.parquet')

# df_mont = df[df['q95']>=0].reset_index(drop=True)
# df_mont = df_mont.sort_values(by='code').reset_index(drop=True)

# subs_new = pd.read_csv('fromMino/base_baciasinc_qinc_calc_sem_ons_sem_climadj.csv')

# df_mont = df_mont[df_mont['code'].isin(subs_new['codigo'])]#.dropna(subset=['qm', 'q95'])
# df_mont.iloc[:, :-2] = df_mont.iloc[:, :-2].fillna(0)
# df_mont.reset_index(drop=True, inplace=True)
# df_mont = df_mont.rename(columns={'nuareamont': 'A'})
# df_mont[['qm', 'q95']] = df_mont[['qm', 'q95']].div(df_mont['A'], axis=0)*1000
# del subs_new

# df_mont.to_parquet('data/subs_attributes_raw.parquet')

# =============================================================================
# Aggregate variables (climate and terrain) to better represent behaviour
# =============================================================================

# df = pd.read_parquet('data/subs_attributes_raw.parquet')

df = pd.read_parquet('data/interim/bho_data_agg.parquet')

subs_new = pd.read_csv('data/external/base_baciasinc_qinc_calc_sem_ons_sem_climadj.csv')

df['has_data'] = df['code'].isin(subs_new['codigo']) #.dropna(subset=['qm', 'q95'])

# qm and q95 at the end
cols = list(df.columns)
cols.remove('qm')
cols.remove('q95')
cols += ['qm', 'q95']

df = df[cols]

# A little bit more processing
df.iloc[:, :-2] = df.iloc[:, :-2].fillna(0)

df = df.rename(columns={'nuareamont': 'A'})
df[['qm', 'q95']] = df[['qm', 'q95']].div(df['A'], axis=0)*1000

del subs_new

# df = df.reset_index(drop=True)

# Function to aggregate monthly variable into monthly avg, min, max
def agg_feat(strend, df_in=df):
    df = df_in.loc[:, df_in.columns.str.endswith(strend)]
    df_agg = pd.DataFrame()
    df_agg['avg'+strend] = df.mean(axis=1)
    df_agg['min'+strend] = df.min(axis=1)
    df_agg['max'+strend] = df.max(axis=1)
    # df_agg['dif'+strend] = df_agg['max'+strend] - df_agg['min'+strend]
    return df_agg

df_prec = agg_feat('_P')
df_temp = agg_feat('_T')
df_et = agg_feat('_ET')
df_pet = agg_feat('_PET')

# Aggregate terrain classes into Savenije simplified classes
df_terrain_simple = pd.DataFrame(columns=['tc_wetland', 'tc_hillslope', 'tc_plateau'])
df_terrain_simple['tc_wetland'] = df['tc_wetland']
df_terrain_simple['tc_hillslope'] = df[['tc_moderate', 'tc_steep', 'tc_extreme']].mean(axis=1)
df_terrain_simple['tc_plateau'] = df[['tc_flat', 'tc_gentle']].mean(axis=1)

# Drop the columns that were aggregated in previous steps
df = df.drop(df.columns[df.columns.str.startswith(('0', '1'))], axis=1)
df = df.drop(df.columns[df.columns.str.startswith('tc_')], axis=1)

df = pd.concat([df_prec, 
                df_temp, 
                df_et, 
                df_pet, 
                df_terrain_simple, 
                df], axis=1)

del df_prec, df_temp, df_et, df_pet, df_terrain_simple

# =============================================================================
# Save for ML model
# =============================================================================
# Remove identification features
# df = df.drop(['lat', 'lon', 'cocursodag', 'nucomptrec', 'L'], axis=1)
# df = df.sample(frac = 1).reset_index(drop=True) # Shuffle values MAKES ALL THE DIFFERENCE IDKW
# df = df.set_index('cobacia')

df.to_parquet('data/processed/data4ml_bho.parquet')

df_gauges = df[df.has_data].reset_index(drop=True).drop('has_data', axis=1) # Comment for whole dataset
df_gauges.to_parquet('data/processed/data4ml_gauges.parquet')





