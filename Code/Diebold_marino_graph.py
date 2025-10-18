
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import diebold_mariano_test, rmse_func

list_of_dfs = [
    [
        'GDP_quarterly_visualization_2009_2019',
        'GDP',
    ],
]

def dm(actual, baseline_resid, model_resid):
    dm_stat, p_val = diebold_mariano_test(actual, baseline_resid, model_resid)
    return [dm_stat, p_val]

results_all = pd.DataFrame()
for csv_name, short_label in list_of_dfs:
    df_all_read = pd.read_csv(f"{csv_name}.csv")
    df_all = df_all_read

    results_all[f"{short_label}_moirai_large"] = (
        dm(df_all["actual"],
           df_all["actual"] - df_all["autoarima"],
           df_all["actual"] - df_all["moirai_large"])
        + [rmse_func(df_all["actual"], df_all["moirai_large"])]
    )

    results_all[f"{short_label}_moirai_base"] = (
        dm(df_all["actual"],
           df_all["actual"] - df_all["autoarima"],
           df_all["actual"] - df_all["moirai_base"])
        + [rmse_func(df_all["actual"], df_all["moirai_base"])]
    )

    results_all[f"{short_label}_LSboost"] = (
        dm(df_all["actual"],
           df_all["actual"] - df_all["autoarima"],
           df_all["actual"] - df_all["LSboost"])
        + [rmse_func(df_all["actual"], df_all["LSboost"])]
    )

    results_all[f"{short_label}_Factor"] = (
        dm(df_all["actual"],
           df_all["actual"] - df_all["autoarima"],
           df_all["actual"] - df_all["Factor"])
        + [rmse_func(df_all["actual"], df_all["Factor"])]
    )

results_all.index = ["DM_stat", "p_value", "rmse"]
dm_long = (
    results_all
    .T
    .reset_index()
    .rename(columns={"index": "models"})
)

dm_long[["DM_stat", "p_value", "rmse"]] = dm_long[["DM_stat", "p_value", "rmse"]].round(4)
dm_long.to_csv("dm_results_moirai_2009_2019.csv", index=False)