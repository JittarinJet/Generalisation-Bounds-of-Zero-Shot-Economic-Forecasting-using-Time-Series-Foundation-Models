import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

datasets = {
    "National GDP (2009Q1 - 2019Q1)": "GDP_quarterly_visualization_2009_2019.csv",
}
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.edgecolor":    "none",
})

title, fpath = next(iter(datasets.items()))
df = pd.read_csv(fpath, parse_dates=["ds"], dayfirst=True)
desired = ["actual", "Moirai_Base", "LSboost", "Factor"]
to_plot = [c for c in desired if c in df.columns]

fig, ax = plt.subplots(figsize=(14, 6))
for col in to_plot:
    ax.plot(df["ds"], df[col], marker="o",
                 markersize=4 if col == "actual" else 3,
                 linewidth=2.5 if col == "actual" else 1.5,
                 label=col)

ax.set_title(title, fontsize=15, weight="bold", pad=15)
ax.set_xlabel("Date", fontsize=15)
ax.set_ylabel("Value", fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.yaxis.set_major_locator(MaxNLocator(nbins="auto"))
ax.yaxis.grid(True, linewidth=0.3, alpha=0.25)
ax.set_axisbelow(True)

y0, y1 = ax.get_ylim()
pad = 0.05 * (y1 - y0)
ax.set_ylim(y0 - pad, y1 + pad)
ax.legend(
    loc="upper left",
    bbox_to_anchor=(0, 1.05),
    frameon=False,
    ncol=len(to_plot),
    fontsize=15
)
fig.tight_layout()
fig.savefig("Comparison_with_RZ.pdf")
plt.show()
