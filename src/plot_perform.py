import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.analyst import Bivar
plt.style.use("seaborn-v0_8")

# -------------- SETUP -------------- #
b_plots = True
compute_metrics = True
suf = ""

lst_models = [
    "MLR",
    "DT",
    "KNN",
    "SVM",
    "GBM",
    "RF"
]

lst_targets = ["qm", "q95"]

# -------------- PERFORMANCE -------------- #
lst_models_perf = []
lst_targets_perf = []
lst_rsq = []
lst_rmse = []
lst_bias = []
for model in lst_models:
    for target in lst_targets:
        f = "data/output/results_raw_{}_{}_k-fold.parquet".format(target, model)
        df = pd.read_parquet(f)
        obs_lbl = "obs"
        pred_lbl = "pred"
        print(pred_lbl)

        lst_models_perf.append(model)
        lst_targets_perf.append(target)
        # compute metrics
        lst_bias.append(Bivar.bias(pred=df[pred_lbl], obs=df[obs_lbl]))
        lst_rsq.append(Bivar.rsq(pred=df[pred_lbl], obs=df[obs_lbl]))
        lst_rmse.append(Bivar.rmse(pred=df[pred_lbl], obs=df[obs_lbl]))

        biv = Bivar(
            df_data=df,
            x_name=obs_lbl,
            y_name=pred_lbl,
            name="{}_{}{}".format(target, model, suf)
        )

        # -------------- error -------------- #
        if b_plots:
            vmax = df[obs_lbl].quantile(1)
            specs = {
                "xlabel": "Obs (l.s^{-1}.km^{-1})",
                "ylabel": "Pred (l.s^{-1}.km^{-1})",
                "xlim": [-0.05*vmax, 1.05*vmax],
                "ylim": [-0.05*vmax, 1.05*vmax],
                "width": 7,
                "height": 7,
                "alpha_xy": 0.25,
                "alpha_e": 0.25,

            }
            biv.view_model(
                model_type="Linear",
                specs=specs,
                folder="docs/figures",
                filename="{}_{}{}".format(target, model, suf),
                show=False
            )

if compute_metrics:
    # -------------- METRICS -------------- #
    # performance metrics
    df_perf = pd.DataFrame(
        {
            "Model": lst_models_perf,
            "Target": lst_targets_perf,
            "Bias": lst_bias,
            "RMSE": lst_rmse,
            "R2": lst_rsq
        }

    )
    df_perf["Bias_abs"] = df_perf["Bias"].abs()
    print(df_perf.to_string())
    # df_perf.to_csv(
    #     "C:/data/ML/performance{}.csv".format(suf),
    #     sep=";",
    #     index=False
    # )

    if b_plots:
        lst_metrics = ["Bias", "RMSE", "R2"]
        for target in lst_targets:
            df_metrics = df_perf.query("Target == '{}'".format(target))
            for metric in lst_metrics:
                ascend = False
                if metric == "R2":
                    ascend = True
                if metric == "Bias":
                    df_metrics = df_metrics.sort_values(
                        by="Bias_abs", ascending=ascend).reset_index(drop=True)
                else:
                    df_metrics = df_metrics.sort_values(
                        by=metric, ascending=ascend).reset_index(drop=True)
                # plot
                plt.figure(figsize=(2, 7))
                plt.barh(
                    df_metrics["Model"],
                    df_metrics[metric],
                    height=0.5,
                    color="dimgray")
                # Adding value labels
                for index, value in enumerate(df_metrics[metric].values):
                    plt.text(value, index+0.3, str(round(value, 2)), ha='center')

                if metric == "Bias":
                    absmax = np.max(np.abs(df_perf.loc[df_perf.Target==target, metric].values))
                    plt.xlim([-absmax*1.5, absmax*1.5])
                elif metric == "R2":
                    plt.xlim([0, 1])
                else:
                    vmax = np.max(df_perf.loc[df_perf.Target==target, metric].values)
                    plt.xlim([0, vmax * 1.5])
                # plt.title("{} {}".format(target, metric))
                plt.tight_layout()

                #plt.show()
                plt.savefig(
                    "docs/figures/{}_{}{}.png".format(target, metric, suf),
                    dpi=300
                )