import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline
import os
import matplotlib.dates as mdates

names = ['GDP', 'Primary_Industries', 'Goods_Producing_Industries', 'Services_Industries']
MODEL = 'chronos-t5-large'  # model size: choose from {'chronos-t5-small', 'chronos-t5-base', 'chronos-t5-large'}
quantile_levels_list = [[0.0, 0.5, 1.0]]
START_POINT = [41]
STEP_H = 1
END_POINT = 0
for name in names:
    df = pd.read_csv(f"datasets/{name}.csv")
    print('df', df)
    END_POINT = len(df)
    for start_point in START_POINT:
        for quantile_levels in quantile_levels_list:
            lows = np.array([])
            medians = np.array([])
            highs = np.array([])
            forecast_indexes = np.array([])
            df_Y_forecasts = pd.DataFrame()
            df_Y_test_all = pd.DataFrame()
            df_Y_train_all = pd.DataFrame()
            for step in range(start_point, END_POINT):
                pipeline = BaseChronosPipeline.from_pretrained(
                    f"amazon/{MODEL}",
                    device_map="cpu",
                    torch_dtype=torch.bfloat16,
                )
                Y_train_df = df[:step]
                df_Y_train_all = df[:step]
                Y_test_df = df[step: step+1]
                df_Y_test_all = pd.concat([df_Y_test_all, Y_test_df], ignore_index=True)
                quantiles, mean = pipeline.predict_quantiles(
                    context=torch.tensor(Y_train_df["Data_value"]),
                    prediction_length=STEP_H,
                    quantile_levels=quantile_levels,
                )
                print('quantiles', quantiles)
                print('mean', mean)
                forecast_index = range(len(Y_train_df), len(Y_train_df) + STEP_H)
                forecast_indexes = np.append(forecast_indexes, forecast_index)
                low, median, high = quantiles[0, :, 0], quantiles[0, :, 1], quantiles[0, :, 2]
                lows = np.append(lows, low)
                medians = np.append(medians, median)
                highs = np.append(highs, high)

            df_Y_test_all.to_csv(f'results/all_steps/{name}_{start_point}_{END_POINT}_{quantile_levels}_{MODEL}_test.csv', index=False)
            pd.DataFrame(medians).to_csv(f'results/all_steps/{name}_{start_point}_{END_POINT}_{quantile_levels}_{MODEL}_median.csv', index=False)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df_Y_train_all["Period"], df_Y_train_all["Data_value"], label="historical data", color='#000c66')
            ax.plot(df_Y_train_all[start_point-1:]["Period"], medians, label="median forecast", color='#CE9DD9')
            ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.tick_params(axis='x', labelrotation=30)
            ax.legend()
            ax.grid()
            plt.savefig(f'results/all_steps/Chronos_{name}_{start_point}_{END_POINT}_{quantile_levels}_{MODEL}.png')
