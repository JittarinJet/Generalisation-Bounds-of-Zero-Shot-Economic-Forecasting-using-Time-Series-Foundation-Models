import pandas as pd

# 1. Load the published metrics
# adjust if the name changes
file_paths = [
    "1999-2024-year_horizon_results.csv",
    "2017-2019-year_horizon_results.csv",
    "2020_2022-year_horizon_results.csv",
    "2023-2024-year_horizon_results.csv",
]

for file_path in file_paths:
    df = pd.read_csv(file_path)
    # 2. Collect the RMSE columns for each sector
    rmse_cols = [
        "National GDP RMSE",
        "Primary Industries RMSE",
        "Goods-Producing Industries RMSE",
        "Services Industries RMSE",
    ]

    # 3. Rank the models sector-by-sector (lower RMSE ⇒ better rank = 1)
    rank_df = df[["Model"]].copy()
    for col in rmse_cols:
        rank_col = col.replace("RMSE", "Rank")
        rank_df[rank_col] = df[col].rank(method="min")

    # 4. Mean rank across the four sectors
    rank_cols = [c for c in rank_df.columns if c.endswith("Rank")]
    rank_df["Mean Rank"] = rank_df[rank_cols].mean(axis=1)

    # 5. Merge and optionally sort, then save
    out = df.merge(rank_df[["Model", "Mean Rank"]], on="Model")
    out = out.sort_values("Mean Rank")

    out.to_csv(f"{file_path}_results_with_rank.csv", index=False)

    print(out[["Model", "Mean Rank"]])
