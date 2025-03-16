import numpy as np
import pandas as pd
import src.process_modelling.functions as mlp


df = pd.read_parquet('data/processed/data4ml_gauges.parquet') # gauges or bho
df = df.loc[:, ~(df==0).all(axis=0)]
df = df.drop(['code', 'g_area', 'g_lat', 'g_lon'], axis=1)

# mlp.plot_correlation_matrix(df)
df = pd.merge(df, results_df, how='right', left_index=True, right_on='station')

# Choose target
# target = ['Q05', 'Q10', 'Q25', 'Q50', 'Q75']
target = ['shape', 'loc', 'scale']
models = ['MLR', 'DT', 'RF']
features = df.columns[:59]

df = df.sample(frac = 1) # Shuffle values

cluster_feature, selected_features = mlp.fs_hcluster(df,
                                                    features, 
                                                    target, 
                                                    cluster_threshold=0.3, 
                                                    link_method='average',
                                                    plot=False)
print(selected_features)
X = df[selected_features].values
y = df[target].values


mlmodel='MLR'
yhat, imps = mlp.model_run(X,
                            y,
                            mlmodel,
                            method='k-fold',
                            )

imps.columns = selected_features


p = 0.5  # e.g. want the flow at CDF=0.90
# flow_pred = df['Q05'].values
flow_obs = genextreme.ppf(p, y[:, 0], loc=y[:, 1], scale=y[:, 2])
flow_pred = genextreme.ppf(p, yhat[:, 0], loc=yhat[:, 1], scale=yhat[:, 2])

mlp.plot_results(flow_obs, flow_pred, imps, f'flow_{p}', mlmodel, savefigs=False)


    
# imps.to_parquet('data/output/imps_'+target+'_'+mlmodel+'_'+method+'.parquet')