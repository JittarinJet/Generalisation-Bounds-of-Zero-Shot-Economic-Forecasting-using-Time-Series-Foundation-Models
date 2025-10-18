import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from util import rmse_func

datasets = {
    "National GDP":               "GDP_visualization_2017_2024.csv",
    "Primary Industries":         "Primary_Industries_visualization_2017_2024.csv",
    "Goods-Producing Industries": "Goods_visualization_2017_2024.csv",
    "Services Industries":        "Services_Industries_visualization_2017_2024.csv",
}

def split_models(df, fpath):
    all_models = [c for c in df.columns if c not in ("ds", "actual")]
    auto = next((c for c in all_models if "autoarima" in c.lower()), None)
    candidates = [c for c in all_models if c != auto]
    result = []
    if "actual" in df.columns:
        rmse = {
            c: rmse_func(df["actual"], df[c]) for c in candidates
        }
        result.append(rmse)
        top3 = sorted(rmse, key=rmse.get)[:3]

    pd.DataFrame(result).to_csv(f'top3_{fpath}.csv', index=False)
    top_group = top3 + ([auto] if auto else [])
    rest_group = [c for c in all_models if c not in top_group]
    return top_group, rest_group

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.edgecolor":    "none",
})

fig1, axes1 = plt.subplots(
    nrows=len(datasets), ncols=1, figsize=(14, 18), sharex=False
)

for ax, (title, fpath) in zip(axes1, datasets.items()):
    df = pd.read_csv(fpath, parse_dates=["ds"], dayfirst=True)
    df = df[:len(df)]
    top_grp, _ = split_models(df, fpath)
    top_grp = (["actual"] if "actual" in df.columns else []) + top_grp

    for col in top_grp:
        if col == "actual":
            ax.plot(
            df["ds"], df['actual'],
            marker="o", markersize=3, linewidth=2.5, label=col
        )
        else:    
            ax.plot(
                df["ds"], df[col],
                marker="o", markersize=3, linewidth=1.2, label=col
            )
    ax.set_title(f'{title} 2017Q1-2024Q3', fontsize=15, weight="bold")
    ax.set_ylabel("", fontsize=15)
    ax.yaxis.set_major_locator(MaxNLocator(nbins="auto"))
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.25)
    ax.set_axisbelow(True)

    # 5 % padding on y-axis for breathing room
    y0, y1 = ax.get_ylim()
    pad = 0.05 * (y1 - y0)
    ax.set_ylim(y0 - pad, y1 + pad)

    ax.legend(
        loc="upper left", bbox_to_anchor=(0, 1.02),
        frameon=False, ncol=len(top_grp), fontsize=13
    )
ax.set_xlabel("Date", fontsize=15)
fig1.tight_layout()
fig1.savefig("top_plus_autoarima_fullwidth.pdf")

# FIGURE 2  : Other models in 2x2 grid
fig2, axes2 = plt.subplots(
    2, 2, figsize=(14, 10), sharex=False, sharey=False
)
axes2 = axes2.flatten()

for ax, (title, fpath) in zip(axes2, datasets.items()):
    df = pd.read_csv(fpath, parse_dates=["ds"], dayfirst=True)
    df = df[:len(df)-1]
    print(df)
    _, rest_grp = split_models(df, fpath)

    for col in rest_grp:
        ax.plot(
            df["ds"], df[col],
            marker="o", markersize=2.5, linewidth=1, label=col
        )

    ax.set_title(f'{title} 2017Q1-2024Q3', fontsize=11, weight="bold")
    ax.set_ylabel("")
    ax.yaxis.set_major_locator(MaxNLocator(nbins="auto"))
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.25)
    ax.set_axisbelow(True)

    # 5 % vertical padding for clarity
    y0, y1 = ax.get_ylim()
    pad = 0.05 * (y1 - y0)
    ax.set_ylim(y0 - pad, y1 + pad)

    # Compact legend inside the panel (smaller font)
    ax.legend(
        loc="upper left", frameon=False,
        fontsize=13, ncol=2, handlelength=1.5
    )

fig2.tight_layout()
fig2.savefig("Appendix_remaining_models_2x2.pdf")

plt.show()
