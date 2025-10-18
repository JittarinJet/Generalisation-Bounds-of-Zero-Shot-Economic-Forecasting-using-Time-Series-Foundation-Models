import pandas as pd, numpy as np, os

FILES = [
    "GDP_visualization_1999_2024.csv",
    "Goods_visualization_1999_2024.csv",
    "Primary_Industries_visualization_1999_2024.csv",
    "Services_Industries_visualization_1999_2024.csv",
]
TARGET = "actual"
OUTDIR = "."

def compute_crps_point(df: pd.DataFrame, target: str = "actual"):
    model_cols = [col for col in df.columns if col not in ["ds", target]]
    per_time = df[["ds", target]].copy()
    for model in model_cols:
        per_time[f"crps_{model}"] = (df[model] - df[target]).abs()

    rows = []
    for model in model_cols:
        vals = per_time[f"crps_{model}"].dropna().values
        if len(vals) == 0:
            continue
        rows.append({
            "model": model,
            "mean_CRPS": float(np.mean(vals)),
            "median_CRPS": float(np.median(vals)),
            "p95_CRPS": float(np.percentile(vals, 95)),
            "p90_CRPS": float(np.percentile(vals, 90)),
            "p85_CRPS": float(np.percentile(vals, 85)),
            "p80_CRPS": float(np.percentile(vals, 80)),
        })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        # Tail-risk convenience metrics
        summary["tail_spread"] = summary["p95_CRPS"] - summary["median_CRPS"]
        summary["upper_tail_steepness"] = summary["p95_CRPS"] - summary["p80_CRPS"]
        summary["mean_rank"] = summary["mean_CRPS"].rank(method="min", ascending=True).astype(int)
        summary = summary.sort_values("mean_CRPS").reset_index(drop=True)
    return per_time, summary

def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col != "ds":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

all_summaries = []
for path in FILES:
    name = os.path.splitext(os.path.basename(path))[0].replace("_visualization_1999_2024", "")
    df = pd.read_csv(path)
    df = clean_numeric(df)

    per_time, summary = compute_crps_point(df, TARGET)
    per_time_out = os.path.join(OUTDIR, f"CRPS_per_timestamp_{name}.csv")
    sum_output = os.path.join(OUTDIR, f"CRPS_summary_{name}.csv")
    per_time.to_csv(per_time_out, index=False)
    summary.to_csv(sum_output, index=False)

    temp = summary.copy()
    temp["sector"] = name
    all_summaries.append(temp)

# Combined summary across all datasets
combined = pd.concat(all_summaries, ignore_index=True)
combined.to_csv(os.path.join(OUTDIR, "CRPS_tail_summary_4_sectors.csv"), index=False)
