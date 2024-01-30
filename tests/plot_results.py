# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 13:56:21 2023

@author: Rafael
"""

# =============================================================================
# Plot importances
# =============================================================================
imps_mean = imps.median()
imps_std = imps.std()
imps_sortindex = imps_mean.argsort()[::-1]
for i in imps_sortindex:
    if imps_mean[i] - 1 * imps_std[i] > 0:
        print(f"{df_new.columns[i]:<16}"
              f"{imps_mean[i]:.3f}"
              f" +/- {imps_std[i]:.3f}")

fig, ax = plt.subplots(1, 1, figsize=(6, 8))
ax.boxplot(
    imps.iloc[:, imps_sortindex[::-1][-12:]],
    vert=False,
    showfliers=False,
    labels=imps.columns[imps_sortindex[::-1][-12:]],
)
ax.set_title('{m}, {t}'.format(m=mlmodel, t=target))
# ax.set_xlim([-0.25,0.75])
fig.tight_layout()
plt.show()
# fig.savefig('figures/permimp_'+target+'_'+mlmodel+'.png', dpi=300)

# =============================================================================
# Stats FOR SEPARATE SPLIT
# =============================================================================
y[y==0] = 0.0001
result[result==0] = 0.0001

def stats(s1, s2):
    rq75 = np.percentile(np.maximum(abs(s1/s2), abs(s2/s1)), 75)
    r2 = 1 - ((s2-s1)**2).sum()/((s2-s2.mean())**2).sum() # == nash
    rmse = ((s1 - s2) ** 2).mean() ** .5
    bias = ((s1-s2).sum())/s2.sum() * 100
    return {'rq75': rq75,
            'r2': r2,
            'rmse': rmse,
            'bias': bias}

# stats_train = stats(result_train, y_train)
# stats_test = stats(result_test, y_test)
stats_loo = stats(result, y)
bias = stats_loo['bias'].round(2)
rmse = stats_loo['rmse'].round(2)
r2 = stats_loo['r2'].round(2)
rq75 = stats_loo['rq75'].round(2)

rq = np.maximum(abs(y/result), abs(result/y))

# # Plot P x q
# fig, ax = plt.subplots(dpi=300, figsize=(5,5))
# ax.scatter(X[:,1], y)
# # ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
# ax.set_xlabel('Precipitation')
# ax.set_ylabel('Discharge')
# # ax.set_xlim([0, np.percentile(X[:,1], 99)])
# # ax.set_ylim([0,np.percentile(y, 99)])
# plt.show()


# Plot observed x predicted
fig, (ax1) = plt.subplots(1, 1, dpi=300, figsize=(5,5))
ax1.scatter(y, result, s=5, alpha=0.5)
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
ax1.set_xlabel('Obs ' + target + ' [l/s]')
ax1.set_ylabel('Pred ' + target + ' [l/s]')
ax1.set_title(mlmodel + ' \n '
             + 'BIAS: ' + str(bias)
             + ', RMSE: ' + str(rmse)
             + ', R²: ' + str(r2)
             # + ', RQ75: ' + str(rq75)
             )
cutlim = np.percentile(y, 99)
ax1.set_xlim([-1, cutlim])
ax1.set_ylim([-1, cutlim])
ax1.set_aspect('equal', 'box')
fig.tight_layout()
plt.show()
# fig.savefig('figures/result_'+target+'_'+mlmodel+'.png', dpi=300)

# =============================================================================
# Uncertainty
# =============================================================================

cov = regression_coverage_score(
    y, y_pis[:, 0, 0], y_pis[:, 1, 0]
)

def get_yerr(y_pred, y_pis):
    return np.concatenate(
        [
            np.expand_dims(y_pred, 0) - y_pis[:, 0, 0].T,
            y_pis[:, 1, 0].T - np.expand_dims(y_pred, 0),
        ],
        axis=0,
    )


y_err = get_yerr(result, y_pis)
int_width = (
    y_pis[:, 1, 0] - y_pis[:, 0, 0]
)
y_err[y_err<0]=0.01


fig, axs = plt.subplots(1, 2, figsize=(10, 5))

class_name = "Residual Score"

axs[0].errorbar(
    y,
    result,
    yerr=y_err,
    alpha=0.5,
    linestyle="None",
)
axs[0].scatter(y, result, s=1, color="black")
axs[0].plot(
    [0, max(max(y), max(result))],
    [0, max(max(y), max(result))],
    "-r",
)
axs[0].set_xlabel('Obs ' + target + ' [l/s]')
axs[0].set_ylabel('Pred ' + target + ' [l/s]')
axs[0].grid()
# axs[0].set_title(f"{class_name} - coverage={cov:.0%}")
cutlim = np.percentile(y, 99)
axs[0].set_xlim([-1, cutlim])
axs[0].set_ylim([-1, cutlim])

xmin, xmax = axs[0].get_xlim() + np.array([0, 0])
ymin, ymax = axs[0].get_ylim() + np.array([0, 0])
axs[1].scatter(y, int_width, marker="+")
axs[1].set_xlabel('Obs ' + target + ' [l/s]')
axs[1].set_ylabel('Prediction interval width' + ' [l/s]')
axs[1].grid()
axs[1].set_xlim([xmin, xmax])
axs[1].set_ylim([ymin, ymax])

fig.suptitle(
    # f"Predicted values with the prediction intervals of level [0.05]"
    f"{mlmodel} {class_name} - coverage={cov:.0%}"
)
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()
# fig.savefig('figures/uncertainty_'+target+'_'+mlmodel+'.png', dpi=300)


# =============================================================================
# Plot errors
# =============================================================================

fig, (ax1) = plt.subplots(1, 1, dpi=300, figsize=(5,5))
ax1.scatter(dft.avg_P, abs(error)/y, s=5, alpha=0.5)
# ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
ax1.set_xlabel('Predictor')
ax1.set_ylabel('error [%]')
# cutlim = np.percentile(y, 99)
ax1.set_xlim([0, np.percentile(dft.avg_P, 99)])
ax1.set_ylim([0, 1])
# ax1.set_aspect('equal', 'box')
fig.tight_layout()
plt.show()













