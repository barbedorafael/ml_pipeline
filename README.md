# ML regression pioeline

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
imps, stats_cv, dfe = mlp.model_run(df, selected_features, target, mlmodel, plot=True)
```
