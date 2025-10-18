import utilsforecast.losses as ufl
from utilsforecast.evaluation import evaluate
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

names = ['GDP', 'Primary_Industries', 'Goods_Producing_Industries', 'Services_Industries', 'GDP_quarterly']
START_POINT = [41]
TEMP_STRAT_POINT = 0
for name in names:
    parameter_case = 'parameters_set_1'
    path = f'datasets/{name}.csv'
    df = pd.read_csv(path, index_col=None)
    df.columns=['unique_id','ds','y']
    df["ds"] = pd.to_datetime(df["ds"])
    actual_df = df
    df.sort_index(inplace=True)
    STEPS = len(df)
    END_POINT = len(df)
    for start_point in START_POINT:
        df_Y_forecasts_all = pd.DataFrame(columns=['unique_id','ds','y'])
        df_Y_test_all = pd.DataFrame(columns=['unique_id','ds','y'])
        df_Y_train_all = pd.DataFrame(columns=['unique_id','ds','y'])
        for step in range(start_point, END_POINT):
            if (step >= TEMP_STRAT_POINT):
                # Past observations
                Y_train_df = df[:step]
                df_Y_train_all = df[:step]
                Y_test_df = df[step: step+1]
                sf = StatsForecast(models=[AutoARIMA(seasonal=True)], freq='QE')
                sf.fit(Y_train_df)
                Y_forecast = sf.forecast(df=Y_train_df, h=1)
                df_Y_test_all = pd.concat([df_Y_test_all, Y_test_df], ignore_index=True)
                df_Y_forecasts_all = pd.concat([df_Y_forecasts_all, Y_forecast], ignore_index=True)
        df_Y_test_all.to_csv(f'results/GDP_AutoArima_results_df_Y_test_{name}_{start_point}_{TEMP_STRAT_POINT}_{END_POINT}.csv', index=False)
        df_Y_forecasts_all.to_csv(f'results/GDP_AutoArima_results_df_Y_forecasts_{name}_{start_point}_{TEMP_STRAT_POINT}_{END_POINT}.csv', index=False)
        fig = sf.plot(actual_df, df_Y_forecasts_all)
        fig.savefig(f'results/GDP_AutoArima_Result_{name}_{start_point}.png')
        