import pandas as pd, numpy as np, os
from scipy import stats

files = {
    'National_GDP': 'GDP_visualization_1999_2024.csv',
    'Primary_Industries': 'Primary_Industries_visualization_1999_2024.csv',
    'Goods_Producing': 'Goods_visualization_1999_2024.csv',
    'Services_Industries': 'Services_Industries_visualization_1999_2024.csv'
}

period_defs = {
    "1999Q3–2024Q4": ("1999-01-01", "2024-01-07"),
    "PreCOVID": ("2017-01-01", "2019-12-31"),
    "COVID": ("2020-01-01", "2021-12-31"),
    "PostCOVID": ("2022-01-01", "2024-01-07")
}

def dm_statistic(errors1, errors2, h=1):
    d = errors1**2 - errors2**2
    T = len(d)
    dm_val = d.mean() / np.sqrt(np.var(d, ddof=0) / T)
    p_val = 2 * stats.t.sf(abs(dm_val), df=T-1)
    return dm_val, p_val

records = []
for sector, path in files.items():
    df = pd.read_csv(path)
    if 'ds' in df.columns:
        df['date'] = pd.to_datetime(df['ds'], dayfirst=True)
    else:
        df['date'] = pd.to_datetime(df['date'])
    ACTUAL = 'actual'
    PERSIST = 'persistence'
    ARIMA = 'autoarima'
    FMS = ['timegpt', 'chronos_small', 'chronos_base', 'chronos_large',
           'moirai_small', 'moirai_base', 'moirai_large']
    
    for col in [PERSIST, ARIMA] + FMS:
        df[f'err_{col}'] = df[col] - df[ACTUAL]
    
    for period, (start, end) in period_defs.items():
        if start is None:
            sub = df
        else:
            sub = df[(df['date'] >= start) & (df['date'] <= end)]
        for fm in FMS:
            _, p_persist = dm_statistic(sub[f'err_{fm}'].values,
                                        sub[f'err_{PERSIST}'].values)
            _, p_arima = dm_statistic(sub[f'err_{fm}'].values,
                                      sub[f'err_{ARIMA}'].values)
            records.append((sector, period, fm, p_persist, p_arima))

res_df = pd.DataFrame(records, columns=["Sector", "Period", "Model",
                                        "p_vs_Persistence", "p_vs_AutoARIMA"])

res_df.to_csv('dm_test_results.csv', index=False)