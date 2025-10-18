import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import mse_func
import random

df = pd.read_csv('GDP_visualization_1999_2024.csv', parse_dates=['ds'], dayfirst=True)
df = df.sort_values('ds').reset_index(drop=True)
label, (start, end) = '1999Q3–2024Q3', ('1999-01-01', '2024-07-01')
seg_full = df[(df['ds'] >= start) & (df['ds'] <= end)]
win_min, win_max, n = 30, 100, len(df)

rand_vars, rand_rmse_b, rand_rmse_l = [], [], []
for index in range(200):
    i0 = random.randint(0, n - win_min - 1)
    i1 = random.randint(i0 + win_min, n - 1)
    window = seg_full.iloc[i0:i1]
    rand_vars.append((window['actual'].var()))
    rand_rmse_b.append(mse_func(window['actual'], window['moirai_base']))
    rand_rmse_l.append(mse_func(window['actual'], window['moirai_large']))

coeff_b = np.polyfit(rand_vars, rand_rmse_b, 1)
coeff_l = np.polyfit(rand_vars, rand_rmse_l, 1)
x_line = np.linspace(min(rand_vars), max(rand_vars), 100)
y_b_line = coeff_b[0] * x_line + coeff_b[1]
y_l_line = coeff_l[0] * x_line + coeff_l[1]
coeff_b = np.polyfit(rand_vars, rand_rmse_b, 1)
coeff_l = np.polyfit(rand_vars, rand_rmse_l, 1)

m_b, c_b = coeff_b
m_l, c_l = coeff_l
fig, ax = plt.subplots(figsize=(14, 8))

# For Moirai Base: keep only those where rand_vars > 3
base_x = [v for v, _ in zip(x_line, y_b_line) if v > 4]
base_y = [rmse_b for v, rmse_b in zip(x_line, y_b_line) if v > 4]

# For Moirai Large: keep only those where rand_vars > 5
large_x = [v for v, _ in zip(x_line, y_l_line) if v > 7]
large_y = [rmse_l for v, rmse_l in zip(x_line, y_l_line) if v > 7]

ax.scatter(rand_vars, rand_rmse_b, alpha=0.3, label='Random Date Range moirai_base')
ax.scatter(rand_vars, rand_rmse_l, alpha=0.3, label='Random Date Range moirai_large')
ax.plot(base_x, base_y, linestyle='--',  label=f'Coefficient {coeff_b[0]:.2f} moirai_base Fit Line')
ax.plot(large_x, large_y, linestyle='-', label=f'Coefficient {coeff_l[0]:.2f} moirai_large Fit Line')
ax.set_title(label, fontsize=20, fontweight='bold', pad=20)
ax.set_xlabel('Actual GDP Variance', fontsize=20)
ax.set_ylabel('MSE', fontsize=20)
ax.legend(fontsize=15, loc='upper left', frameon=True, borderpad=1)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.grid(True, linestyle=':', alpha=0.5)
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

plt.tight_layout()
plt.savefig('GDP_RMSE_variance_comparison_single.pdf')
plt.show()
