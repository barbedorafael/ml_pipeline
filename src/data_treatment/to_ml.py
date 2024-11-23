# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:54:40 2023

@author: RafBar
"""

import numpy as np
import pandas as pd

df = pd.read_parquet('data/interim/bho_data_agg.parquet')

gauges = pd.read_csv('data/external/base_baciasinc_qinc_calc_sem_ons_sem_climadj.csv')

has_data = df['code'].isin(gauges['codigo']) #.dropna(subset=['qm', 'q95'])
df.loc[~has_data, ['code', 'qm', 'q95']] = [0, np.nan, np.nan]

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

del gauges

# df = df.reset_index(drop=True)

# Function to aggregate monthly variable into monthly avg, min, max
def agg_feat(strprefix, df_in=df):
    df = df_in.loc[:, df_in.columns.str.endswith('_{}'.format(strprefix))]
    df_agg = pd.DataFrame()
    df_agg['{}_avg'.format(strprefix)] = df.mean(axis=1)
    df_agg['{}_min'.format(strprefix)] = df.min(axis=1)
    df_agg['{}_max'.format(strprefix)] = df.max(axis=1)
    # df_agg['dif'+strend] = df_agg['max'+strend] - df_agg['min'+strend]
    return df_agg

df_prec = agg_feat('P')
df_temp = agg_feat('T')
df_et = agg_feat('ET')
df_pet = agg_feat('PET')

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


df = df.rename(columns={
                        'P_avg': 'p_avg',
                        'P_min': 'p_min',
                        'P_max': 'p_max',
                        'T_avg': 't_avg',
                        'T_min': 't_min',
                        'T_max': 't_max',
                        'ET_avg': 'et_avg',
                        'ET_min': 'et_min',
                        'ET_max': 'et_max',
                        'PET_avg': 'pet_avg',
                        'PET_min': 'pet_min',
                        'PET_max': 'pet_max',
                        'tc_wetland': 'tc_wtl',
                        'tc_hillslope': 'tc_hls',
                        'tc_plateau': 'tc_plt',
                        'soilclay': 'sc_cly',
                        'soilorga': 'sc_org',
                        'soilsand': 'sc_snd',
                        'soilwate': 'sc_wtr',
                        'elv_avg': 'tp_elv',
                        'slp_avg': 'tp_slp',
                        'hnd_avg': 'tp_hnd',
                        'dd': 'tp_dd',
                        'A': 'tp_ua',
                        'GW': 'ws_gw',
                        'RS': 'ws_rs',
                        'SM': 'ws_sm',
                        'SW': 'ws_sw',
                        'TWS': 'ws_tt',
                        'lc_forest': 'lc_fr',
                        'lc_grassland': 'lc_gr',
                        'lc_agriculture': 'lc_ag',
                        'lc_permeable': 'lc_pr',
                        'lc_water': 'lc_wt',
                        'st_Cl': 'st_cl',
                        'st_SiCl': 'st_sicl',
                        'st_SaCl': 'st_sacl',
                        'st_ClLo': 'st_cllo',
                        'st_SiClLo': 'st_sicllo',
                        'st_SaClLo': 'st_sacllo',
                        'st_Lo': 'st_lo',
                        'st_SiLo': 'st_silo',
                        'st_SaLo': 'st_salo',
                        'st_Si': 'st_si',
                        'st_LoSa': 'st_losa',
                        'st_Sa': 'st_sa',
                        'lt_Su': 'lt_su',
                        'lt_Vb': 'lt_vb',
                        'lt_Ss': 'lt_ss',
                        'lt_Pb': 'lt_pb',
                        'lt_Sm': 'lt_sm',
                        'lt_Sc': 'lt_sc',
                        'lt_Va': 'lt_va',
                        'lt_Mt': 'lt_mt',
                        'lt_Pa': 'lt_pa',
                        'lt_Vi': 'lt_vi',
                        'lt_Wb': 'lt_wb',
                        'lt_Py': 'lt_py',
                        'lt_Pi': 'lt_pi',
                        'lt_Ev': 'lt_ev',
                        'lt_Nd': 'lt_nd',
                        'lt_Ig': 'lt_ig',
                    })

df = df.drop(['lat', 'lon', 'L', 'nucomptrec', 'cocursodag', 'cobacia'], axis=1)

# =============================================================================
# Save for ML model

# Remove identification features
# df = df.drop(['lat', 'lon', 'cocursodag', 'nucomptrec', 'L'], axis=1)
# df = df.sample(frac = 1).reset_index(drop=True) # Shuffle values MAKES ALL THE DIFFERENCE IDKW
# df = df.set_index('cobacia')

df.to_parquet('data/processed/data4ml_bho.parquet')

df_gauges = df[has_data]
df_gauges.to_parquet('data/processed/data4ml_gauges.parquet')





