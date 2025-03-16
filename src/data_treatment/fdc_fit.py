from scipy.stats import genextreme, lognorm
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gev_cdf(params, x):
    """
    Wrapper around scipy's genextreme.cdf
    params: (shape, loc, scale) = (c, mu, sigma)
    x: array-like of flows
    """
    c, loc, scale = params
    return genextreme.cdf(x, c, loc=loc, scale=scale)

def objective(params, flows, cdf_probs):
    """
    Sum of squared errors between the empirical cdf_probs 
    and the GEV cdf of flows with 'params'.
    flows, cdf_probs must be same length.
    """
    cdf_model = gev_cdf(params, flows)
    sse = np.sum((cdf_model - cdf_probs)**2)
    return sse

def fit_gev_to_row(flow_values, cdf_probs):
    """
    Given array of flows for one station, and the matching cdf_probs,
    find the best-fit GEV parameters. Return (shape, loc, scale, SSE).
    """
    # Initial guess for shape, loc, scale
    c_init   =  0.0
    mu_init  = np.mean(flow_values)
    sigma_init = max(1e-3, np.std(flow_values))
    init_params = np.array([c_init, mu_init, sigma_init])

    # Let's define some bounds to keep scale>0
    bounds = [(-2, 2),  (0, None),  (1e-6, None)]
    
    # Minimize SSE
    res = minimize(objective, x0=init_params, 
                   args=(flow_values, cdf_probs),
                   method='L-BFGS-B', bounds=bounds)

    if res.success:
        c_opt, mu_opt, sigma_opt = res.x
        sse = res.fun
        return c_opt, mu_opt, sigma_opt, sse
    else:
        # Return some indicator of failure
        return np.nan, np.nan, np.nan, np.nan


df = pd.read_parquet('data/processed/data4ml_gauges.parquet') # gauges or bho
dflows = df.loc[:, df.columns.str.match(r'^Q\d\d$')]
dflows = dflows.iloc[:, :-9]
results = []
cdf_probs = np.arange(0.01, 0.91, 0.01).round(2)
for station_name, row in dflows.iterrows():
    # row is a Series of flow values for columns df.columns
    # Convert to a numpy array
    flow_values = row.values  # shape (n_cols,)

    # Fit GEV
    c_opt, mu_opt, sigma_opt, sse = fit_gev_to_row(flow_values, cdf_probs)

    # Compute additional metrics or store results
    results.append({
        'station': station_name,
        'shape': c_opt,
        'loc': mu_opt,
        'scale': sigma_opt,
        'sse': sse
    })

results_df = pd.DataFrame(results)

for station_name, row_data in dflows.iterrows():
    # -- Retrieve the flows (empirical FDC points) --
    # row_data is a Series with len == number of columns in df
    flows_empirical = row_data.values

    # -- Retrieve fitted parameters (shape, loc, scale) from results_df --
    # 1) If your results_df uses station_name as index:
    if station_name in results_df.index:
        c_opt  = results_df.loc[station_name, 'shape']
        mu_opt = results_df.loc[station_name, 'loc']
        sc_opt = results_df.loc[station_name, 'scale']
    else:
        # 2) If results_df has station names in a column, we match them that way:
        match_row = results_df[results_df['station'] == station_name]
        # if len(match_row) == 0:
        #     print(f"No matching station '{station_name}' in results_df.")
        #     continue
        c_opt  = match_row.iloc[0]['shape']
        mu_opt = match_row.iloc[0]['loc']
        sc_opt = match_row.iloc[0]['scale']

    # -- Generate a smooth curve for the fitted distribution --
    # For plotting, let's make ~200 points in [0,1], or align with [0.01..0.99].
    cdf_grid = np.linspace(0.001, 0.999, 200)
    # invert the GEV cdf using ppf
    flows_fitted = genextreme.ppf(cdf_grid, c_opt, loc=mu_opt, scale=sc_opt)

    # -- Plot --
    plt.figure(figsize=(6,4))
    # Plot empirical points. Note that cdf_probs is the x-axis, flows_empirical is the y-axis
    # or vice versa, depending on how you prefer FDC orientation.
    # Usually, in FDC plots, we place probability on the x-axis, flow on the y-axis:
    plt.plot(cdf_probs, flows_empirical, 'o', label='Empirical FDC')

    # Plot the fitted GEV curve
    plt.plot(cdf_grid, flows_fitted, '-', label='Fitted GEV')

    plt.ylim([0, np.quantile(flows_empirical, 0.99)])

    plt.title(f"Station: {station_name}")
    plt.xlabel("CDF Probability")
    plt.ylabel("Flow")
    plt.legend()
    plt.tight_layout()

    plt.show()


results_df=results_df.loc[results_df.sse<1]