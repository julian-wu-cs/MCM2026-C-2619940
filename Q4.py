# Q4.py
# Dynamic weighting + bottom-two judges save
# Strictly uses:
#   Data/2026_MCM_Problem_C_Data.csv
#   outputs/Q1/tables/Q1_1.csv

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from project_paths import get_data_path, get_out_dir, ensure_dir

# 1.  Q4.py 
DEFAULT_Q1_CSV = get_out_dir("Q1", None) / "tables" / "Q1_1.csv"

ap = argparse.ArgumentParser(description="Q4")
ap.add_argument("--data", type=str, default=None, help="Path to data csv")
ap.add_argument("--out", type=str, default=None, help="Output directory")
args = ap.parse_args()

OFFICIAL_CSV = get_data_path(args.data)
Q1_CSV = DEFAULT_Q1_CSV
OUT_DIR = ensure_dir(get_out_dir("Q4", args.out))
TABLES_DIR = ensure_dir(OUT_DIR / "tables")

W_MIN  = 0.35
W_MAX  = 0.70
KAPPA  = 0.02
DELTA  = 0.02
USE_BOTTOM_TWO_SAVE = True

if not OFFICIAL_CSV.exists():
    raise FileNotFoundError(f"CSV not found: {OFFICIAL_CSV}")
if not Q1_CSV.exists():
    raise FileNotFoundError(f"Q1 table not found: {Q1_CSV}")

official = pd.read_csv(str(OFFICIAL_CSV))
q1 = pd.read_csv(str(Q1_CSV))

# 4.  judge_total  
judge_cols = [c for c in official.columns if "judge" in c and "week" in c]

records = []
for _, row in official.iterrows():
    season = row["season"]
    name = row["celebrity_name"]
    for c in judge_cols:
        if pd.notna(row[c]):
            week = int(c.split("_")[0].replace("week", ""))
            records.append((season, week, name, row[c]))

judge_long = pd.DataFrame(
    records, columns=["season", "week", "celebrity_name", "score"]
)

judge_total = (
    judge_long
    .groupby(["season", "week", "celebrity_name"], as_index=False)
    .agg(judge_total=("score", "sum"))
)

# 5.  Q1 fan vote 
if "fan_share_sd" not in q1.columns:
    q1["fan_share_sd"] = (
        q1["fan_p95"] - q1["fan_p05"]
    ) / (2 * 1.6448536)

q1 = q1.rename(columns={"fan_pct_mean": "fan_share_mean"})

df = pd.merge(
    judge_total,
    q1[["season", "week", "celebrity_name",
        "fan_share_mean", "fan_share_sd"]],
    on=["season", "week", "celebrity_name"],
    how="inner"
)

rows = []
elim_log = []

for (season, week), g in df.groupby(["season", "week"]):
    g = g.copy()

    g["J_share"] = g["judge_total"] / g["judge_total"].sum()
    g["F_share"] = g["fan_share_mean"] / g["fan_share_mean"].sum()

    U_t = g["fan_share_sd"].mean()
    w_t = W_MIN + (W_MAX - W_MIN) * U_t / (U_t + KAPPA)

    g["U_t"] = U_t
    g["w_t"] = w_t
    g["C_score"] = w_t * g["J_share"] + (1 - w_t) * g["F_share"]

    g = g.sort_values("C_score")

    eliminated = g.iloc[0]["celebrity_name"]
    saved = ""

    if USE_BOTTOM_TWO_SAVE and len(g) >= 2:
        b1, b2 = g.iloc[0], g.iloc[1]
        if b2["J_share"] - b1["J_share"] > DELTA:
            eliminated = b1["celebrity_name"]
            saved = b2["celebrity_name"]

    elim_log.append({
        "season": season,
        "week": week,
        "U_t": U_t,
        "w_t": w_t,
        "eliminated": eliminated,
        "saved_if_any": saved
    })

    rows.append(g)

week_table = pd.concat(rows, ignore_index=True)

placements = []

for season, g in week_table.groupby("season"):
    alive = set(g["celebrity_name"].unique())
    elim_order = []

    for _, r in pd.DataFrame(elim_log).query(
        "season == @season"
    ).sort_values("week").iterrows():
        if r["eliminated"] in alive:
            elim_order.append(r["eliminated"])
            alive.remove(r["eliminated"])

    elim_order = list(reversed(elim_order))
    for i, name in enumerate(elim_order, 1):
        placements.append((season, name, i))

predicted = pd.DataFrame(
    placements,
    columns=["season", "celebrity_name", "predicted_placement"]
)

metrics = []

for season, g in week_table.groupby("season"):
    tech = g.groupby("celebrity_name")["J_share"].sum()
    real = official.query(
        "season == @season"
    ).set_index("celebrity_name")["placement"]

    rho, _ = spearmanr(
        tech.loc[real.index], real
    )

    metrics.append({
        "season": season,
        "tech_fairness_spearman": rho
    })

metrics_df = pd.DataFrame(metrics)

week_table.to_csv(
    str(TABLES_DIR / "week_table_dynamic.csv"),
    index=False
)

pd.DataFrame(elim_log).to_csv(
    str(TABLES_DIR / "elimination_log_dynamic.csv"),
    index=False
)

predicted.to_csv(
    str(TABLES_DIR / "predicted_placements_dynamic.csv"),
    index=False
)

metrics_df.to_csv(
    str(TABLES_DIR / "metrics_dynamic.csv"),
    index=False
)

with open((TABLES_DIR / "run_config.json"), "w") as f:
    json.dump({
        "w_min": W_MIN,
        "w_max": W_MAX,
        "kappa": KAPPA,
        "delta": DELTA,
        "bottom_two_save": USE_BOTTOM_TWO_SAVE
    }, f, indent=2)

print(f"Q4 finished. Results saved to {str(TABLES_DIR)}/")


