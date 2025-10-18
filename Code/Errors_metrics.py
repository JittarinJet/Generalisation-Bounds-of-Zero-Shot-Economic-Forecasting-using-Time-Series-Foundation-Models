import pandas as pd
from util import mae_func, rmse_func, smape_func, mase_func
import numpy as np

def metrics_frame(df, start, end, horizon=1):
    seg = df.loc[start:end].copy()
    actual = seg['actual'].values
    print(seg)
    results = {}
    for col in seg.columns.drop(['actual']):
        pred = seg[col].values
        mae = mae_func(actual, pred),
        rmse = rmse_func(actual, pred),
        smape = smape_func(actual, pred),
        mase = mase_func(actual, pred, m=horizon)
        print('mae', mae)
        results[col] = {
            'MAE' : f'{mae[0]:.2f}',
            'RMSE': f'{rmse[0]:.2f}',
            'SMAPE': f'{smape[0]:.2f}',
            'MASE' : f'{mase:.2f}',
        }
    return pd.DataFrame(results).T

list_of_dfs = [
    [
        'GDP_visualization_1999_2024',
        'GDP',
    ],
    [
        'Goods_visualization_1999_2024',
        'Goods-Producing Industries'
    ],
    [
        'Primary_Industries_visualization_1999_2024',
        'Primary_Industries'
    ],
    [
        'Services_Industries_visualization_1999_2024',
        'Services Industries'
    ],
]
for list_of_df in list_of_dfs:

    df = pd.read_csv(f'{list_of_df[0]}.csv', parse_dates=['ds'])
    df.set_index('ds', inplace=True)
    p1 = metrics_frame(df, '2017-01-01', '2019-12-31')
    p2 = metrics_frame(df, '2020-01-01', '2022-12-31')
    p3 = metrics_frame(df, '2023-01-01', '2024-01-07')
    p4 = metrics_frame(df, '1999-01-01', '2024-01-07')
    
    p1.to_csv(f'{list_of_df[1]}_metrics_2017_2019.csv')
    p2.to_csv(f'{list_of_df[1]}_metrics_2020_2022.csv')
    p3.to_csv(f'{list_of_df[1]}_metrics_2023_2024.csv')
    p4.to_csv(f'{list_of_df[1]}_metrics_1999_2024.csv')