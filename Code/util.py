import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dieboldmariano import dm_test

def smape_func(y_true, y_pred):
    smape = 1/len(y_true) * np.sum( 2 * np.abs(y_pred-y_true)/(np.abs(y_true) + np.abs(y_pred)))
    return smape

def mae_func(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    return np.mean(diff)

def rmse_func(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def rmae_func(y_true, y_pred):
    return mae_func(y_true, y_pred) / np.mean(np.abs(y_true))

def mse_func(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mase_func(y_true, y_pred, m=1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    if n != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length.")
    mae_forecast = np.mean(np.abs(y_true - y_pred))
    abs_diff_naive = np.abs(y_true[m:] - y_true[:-m])
    mae_naive = np.mean(abs_diff_naive)
    if mae_naive == 0:
        return np.nan

    mase = mae_forecast / mae_naive
    return mase

def diebold_mariano_test(actual, e1, e2):
    dm_test(actual, e1, e2)
    return dm_test(actual, e1, e2)
