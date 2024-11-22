# ML regression pipeline

```python
import pandas as pd
import ml_pipeline.functions as mlp

df = pd.read_parquet('data/data4ml.parquet')
df = df.sample(frac = 1) # Shuffle values MAKES ALL THE DIFFERENCE IDKW

# Choose target (qm or q95)
target = 'q95' # q95 or qm

# Select features for modelling based on hyerarchical clustering
cluster_feature, selected_features = mlp.feature_selection(df, target, cluster_threshold=0.4, plot=False)


# Chose ML model 
mlmodel = 'SVM' # 'MLR', 'DT', 'KNN', 'SVM', 'GBM', 'RF'
y, result, imps = mlp.model_run(df,
				selected_features,
				target,
				mlmodel,
				method=method,
				)

# Plot results
mlp.plot_results(result, y, imps, target, mlmodel, savefigs=True)

# Creating the DataFrame with specified columns and errors
dfe = pd.DataFrame(index=df.index)
dfe[target + '_obs'] = y
dfe[target + '_pred'] = result
dfe[target + '_error'] = y - result
```
