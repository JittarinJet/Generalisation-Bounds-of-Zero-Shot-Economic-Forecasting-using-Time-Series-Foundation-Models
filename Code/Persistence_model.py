import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from random import random
from util import smape_func, mae_func, mase_func, rmse_func

names = ['GDP', 'Primary_Industries', 'Goods_Producing_Industries', 'Services_Industries']

START_POINT = [41]
for name in names:
    parameter_case = 'parameters_set_1'
    path = f'datasets/{name}.csv'
    df = pd.read_csv(path, index_col=None)
    df.columns=['unique_id','ds','y']
    df["ds"] = pd.to_datetime(df["ds"])
    END_POINT = len(df) 
    for start_point in START_POINT:
        Y_train_df = df[:start_point]
        Y_test_df = df[start_point:]
        Forecast_df = Y_test_df.copy()
        Forecast_df['y'] = Y_test_df['y'].shift(1)
        plt.figure(figsize=(10, 6))
        plt.plot(df["ds"], df["y"], label='Actual (All Data)', color='#000c66')
        plt.plot(Y_test_df["ds"], Forecast_df["y"], label='Prediction', color='orange')
        plt.title(f'Persistence Model: {name}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        Forecast_df.to_csv(f'results/Forecast_df_{name}_{parameter_case}_{start_point}_{END_POINT}.csv')
        Y_test_df.to_csv(f'results/Y_test_df_{name}_{parameter_case}_{start_point}_{END_POINT}.csv')
        plt.savefig(f'results/GDP_Persistence_model_{name}_{parameter_case}_{start_point}_{END_POINT}.png')
        print('Actual_df', df)
        print('Y_test_df', Y_test_df)
        print(Y_test_df['y'].values.shape, Forecast_df['y'].values.shape)
        test_values = Y_test_df[1:]
        forecast_values =  Forecast_df[1:]
        print('test_values', test_values)
        mae = mae_func(test_values['y'].values, forecast_values['y'].values)
        print(f"MAE: {mae:.4f}")
        rmse = rmse_func(test_values['y'].values, forecast_values['y'].values)
        print(f"rmse: {rmse:.4f}")
        smape = smape_func(test_values['y'].values, forecast_values['y'].values)
        print(f"SMAPE:' {smape:.4f}")
        mase = mase_func(test_values['y'].values, forecast_values['y'].values)
        print(f"MASE:' {mase:.4f}")

        printout = pd.DataFrame({
            'MAE': [f"{mae:.2f}"],
            'RMSE': [f"{rmse:.2f}"],
            'SMAPE': [f"{smape:.2f}"],
            'MASE': [f"{mase:.2f}"]
        })
        printout.to_csv(f'results/Persistence_model_results_{name}_{parameter_case}_{start_point}_{END_POINT}.csv', index=False)
