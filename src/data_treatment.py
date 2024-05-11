# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 11:57:28 2021

@author: Rafael
"""

# =============================================================================
# Preprocess the input files
# =============================================================================
# import numpy as np
# import pandas as pd
# import geopandas as gpd
# import ast

# bho_dir = r'D:\rafbar\OneDrive\PRH\Brasil_data\BHO_database\BHO_2017_v_01_05_5k\\'
# trecs = gpd.read_file(bho_dir + 'geoft_bho_2017_5k_trecho_drenagem.gpkg',
#         include_fields = ['cotrecho', 'cobacia', 'cocursodag', 'nutrjus', 
#                          'nuareacont', 'nuareamont', 'nucomptrec', 'nustrahler',
#                          'dedominial', 'geometry'])

# qdata = gpd.read_file('fromMino/base_bho_baciasinc_qinc_sem_climadj.gpkg',
#        include_fields=['cotrecho', 'posto', 'p1_qmca', 'p1_q95ca'])
# gaugelocs = gpd.read_file('fromMino/base_postos_ana.gpkg',
#             include_fields=['codigo', 'area_km2', 'latitude', 'longitude'])

# subs = pd.merge(qdata.drop('geometry', axis=1), gaugelocs.drop('geometry', axis=1),
#                 how='left',
#                 left_on='posto', right_on='codigo')
# subs = subs.drop('codigo', axis=1)
# subs = subs.rename({'posto': 'code',
#                     'p1_qmca': 'qm',
#                     'p1_q95ca': 'q95',
#                     'area_km2': 'g_area',
#                     'latitude': 'g_lat',
#                     'longitude': 'g_lon'},
#                    axis=1)

# df_work = pd.merge(trecs.drop('geometry', axis=1), subs, on='cotrecho')

# gdf = gpd.GeoDataFrame(df_work, geometry=trecs.geometry)

# gdf.to_parquet('data/input/bho_flow_data.parquet')

# =============================================================================
# Collect attributes
# =============================================================================
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

# datageo = gpd.GeoDataFrame(data.copy(), geometry=subs.geometry)
# datageo.to_file('GIS/bho_data_all.gpkg', driver='GPKG')

# subs[['p_avg', 'p_min', 'p_max']] = p_df[['p_avg', 'p_min', 'p_max']]

# subs.to_file('GIS/bho_attributes_flow.shp')

# p_file = 'GIS/P_annual_mean_1000.tif'
# p_rate = zonal_stats(
#             polys['geometry'], 
#             p_file, all_touched=True, stats='mean')
# p_rate_df = pd.DataFrame(p_rate)#.fillna(0)
# polys['Py_mean'] = p_rate_df

# searchC = 'GIS/*' + 'LC_rm' + '*.tif'
# lc_files = glob.glob(searchC)

# for lc_file in lc_files:
#     lc_rate = zonal_stats(
#                 polys['geometry'], 
#                 lc_file, all_touched=True, categorical=True)
# lc_rate_df = pd.DataFrame(lc_rate)#.fillna(0)
# lc_rate_df.index = polys.index
# lc_rate_df.to_csv('data/lc_rate_polys.csv', index=False)

# polys['LC'] = lc_rate_df

# lc_rate_df = pd.read_csv('data/lc_rate_polys.csv')
# lc_rate_df = lc_rate_df[['1', '2', '3', '4', '5']]
# lc_psum = lc_rate_df.sum(axis=1)
# lc_rate_df = (lc_rate_df.div(lc_psum, axis=0) * 100).round(2)

# polys['LC01'] = lc_rate_df['1']
# polys['LC02'] = lc_rate_df['2']
# polys['LC03'] = lc_rate_df['3']
# polys['LC04'] = lc_rate_df['4']
# polys['LC05'] = lc_rate_df['5']



# df_work['P_mean'] = polys['Py_mean']
# df_work['plat_rate'] = polys['Plateau_ra']
# df_work['hill_rate'] = polys['Hillslope_']
# df_work['wetl_rate'] = polys['Wetland_ra']
# df_work['Dd'] = polys['Dd']
# df_work['LC01'] = polys['LC01']
# df_work['LC02'] = polys['LC02']
# df_work['LC03'] = polys['LC03']
# df_work['LC04'] = polys['LC04']
# df_work['LC05'] = polys['LC05']

# =============================================================================
# Weighted average for subbasins (for Qinc)
# =============================================================================
# import pandas as pd
# import geopandas as gpd

# df_work = gpd.read_file('GIS/bho_regionalization_test.shp')

# def w_avg(df, values, weights):
#     d = df[values]
#     w = df[weights]
#     dw = d.multiply(w, axis=0)
#     return (dw).sum() / w.sum()


# df_subs = df_work.groupby('sub').apply(w_avg, ['P_mean', 'plat_rate', 'hill_rate', 'wetl_rate', 'Dd'],
#                                         'nuareacont')
# df_subs = df_subs.loc[df_subs.index>0, :]
# data_sub = df_work.dropna(subset=['qm']).set_index('sub', drop=True)
# df_subs['A'] = df_work.groupby('sub').sum()['nuareacont']
# df_subs['qm'] = data_sub['qm_inc']
# df_subs['q95'] = data_sub['q95_inc']

# df_subs = df_subs[['A', 'P_mean', 'plat_rate', 'hill_rate', 'wetl_rate', 'Dd', 'qm', 'q95']]
# df_subs.to_csv('data/subs_attributes_qinc.csv')

# =============================================================================
# Weighted average for upstream cobasins (for global Q)
# =============================================================================
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




#### FOR VALIDATION (only where observed data available)

def upstream_domain(x, df=df):
    trec_out = x['cobacia']#.iloc[0] # Trecho mais de jusante
    wc_out = x['cocursodag']#.iloc[0] # Curso d'agua desse trecho
    
    df_filtered = df[(df['cocursodag'].str.startswith(wc_out)) # Filtra cursos d'agua
                     & (df['cobacia']>=trec_out)] # Filtra cobacias a montante

    return df_filtered

# Drop qinc and nodata values
df_subs = df[df['qm']>0].iloc[:, :-2].reset_index(drop=True)

# df_subs = df.apply(lambda row:
#               w_avg(upstream_domain(row), ['P_mean', 'plat_rate', 'hill_rate', 'wetl_rate', 'Dd'],
#                                         'nuareacont'), axis=1)


only_att = ~df_subs.columns.isin(['cobacia', 'cocursodag', 'cotrecho', 'nuareamont',
                                'sub', 'posto', 
                                'qm', 'q95', 'qm_inc', 'q95_inc'])
df_mont = df_subs.copy()
df_mont.loc[:, only_att] = 0
start_time = time.time()
for key, i in df_subs.iterrows():
    trecs_up = upstream_domain(i)
    wavg = trecs_up[values_wavg]#.to_numpy()
    w = trecs_up['nuareacont']#.to_numpy()
    att_avg = np.ma.average(wavg, axis=0, weights=w).filled(np.nan)
    df_mont.loc[key, values_wavg] = att_avg
    df_mont.loc[key, 'A'] = trecs_up['nuareacont'].sum()
    print(key/df_subs.shape[0])
df_mont = df_mont.dropna(axis=1).reset_index(drop=True)
end_time = time.time()
print('Time to process: ' + str(((end_time - start_time)/60)) + ' min')

# df_mont.to_pickle('data/subs_attributes.pkl')

df_subs.groupby(upstream_domain)

df_subs = df_work.groupby('sub').apply(w_avg, ['P_mean', 'plat_rate', 'hill_rate', 'wetl_rate', 'Dd'],
                                        'nuareacont')
df_subs = df_subs.loc[df_subs.index>0, :]
data_sub = df_work.dropna(subset=['qm']).set_index('sub', drop=True)
df_subs['A'] = data_sub['nuareamont']
df_subs['qm'] = data_sub['qm']
df_subs['q95'] = data_sub['q95']

# df_subs = df_subs[['A', 'P_mean', 'plat_rate', 'hill_rate', 'wetl_rate', 'Dd', 'qm', 'q95']]















