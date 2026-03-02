# 2026 MCM/ICM Problem C - Data With The Stars

## Project Overview
This repository contains a full code pipeline for **2026 MCM/ICM Problem C**, including four analytical tasks (`Q1.py`-`Q4.py`) and four visualization scripts (`visual1.py`-`visual4.py`).

The project estimates hidden fan vote shares, compares voting combination rules, analyzes influential factors with machine learning, and proposes an alternative elimination system. All scripts are organized around a unified path utility (`project_paths.py`) and write outputs to `outputs/`.

## Problem C Overview
Based on `Data/2026_MCM_Problem_C.pdf`, the core tasks are:

1. Infer weekly fan votes (unknown in real DWTS data) and evaluate whether inferred votes are consistent with observed eliminations.
2. Quantify uncertainty of fan vote estimates by contestant/week.
3. Compare the two official score-combination methods across seasons:
   - Rank-based combination.
   - Percent-based combination.
4. Examine controversy cases and evaluate the impact of adding the bottom-two judges-save mechanism.
5. Recommend a voting method for future seasons.
6. Analyze how pro dancers and celebrity attributes affect outcomes (fan and judge channels).
7. Propose a new system balancing fairness and entertainment.

### Data fields (from problem statement)
`2026_MCM_Problem_C_Data.csv` includes (not limited to):
- `celebrity_name`, `ballroom_partner`
- `celebrity_industry`, `celebrity_homestate`, `celebrity_homecountry/region`, `celebrity_age_during_season`
- `season`, `results`, `placement`
- `weekX_judgeY_score` (judge score per week/judge)

Notes from the problem statement reflected in code assumptions:
- Seasons covered: 1-34.
- Number of contestants and weeks varies by season.
- Missing values exist for unused judges/weeks.
- Zeros may represent post-elimination weeks for contestants.

## Method Pipeline (Q1 -> Q4)
1. `Q1.py`: infer fan-share posterior distributions and uncertainty metrics week-by-week.
2. `Q2.py`: compare rank/percent/judge-save methods using Q1 outputs + raw judge data.
3. `Q3.py`: model fan and judge shares with Random Forest + interpretation outputs.
4. `Q4.py`: simulate a dynamic weighting elimination rule and evaluate fairness-oriented metrics.
5. `visual1.py`-`visual4.py`: generate publication-style figures from each question's tables.

## Repository Structure And File Guide

```text
.
|-- Data/
|   |-- 2026_MCM_Problem_C.pdf
|   |-- 2026_MCM_Problem_C_Data.csv
|-- outputs/
|   |-- Q1/{tables,figs}
|   |-- Q2/{tables,figs}
|   |-- Q3/{tables,figs}
|   `-- Q4/{tables,figs}
|-- 2619940.pdf
|-- project_paths.py
|-- Q1.py
|-- Q2.py
|-- Q3.py
|-- Q4.py
|-- visual1.py
|-- visual2.py
|-- visual3.py
`-- visual4.py
```

### `project_paths.py` (path unification)
Purpose:
- Centralizes default project root, input data path, and output directory construction.

Mechanism:
- `get_data_path(data_arg)`:
  - if `--data` is provided, uses that path directly;
  - otherwise defaults to `Data/2026_MCM_Problem_C_Data.csv`.
- `get_out_dir(q, out_arg)`:
  - if `--out` is provided, uses that path directly;
  - otherwise defaults to `outputs/<Qk>`.
- `ensure_dir(path)` creates directories recursively before writing outputs.

Reproducibility implication:
- All `Q*.py` and `visual*.py` scripts rely on this module for consistent I/O behavior.

### Per-file audit (purpose, input, flow, output)

#### `Q1.py`
Purpose:
- Core inverse modeling to estimate hidden fan vote shares and uncertainty per contestant-week.

Inputs:
- Data CSV from `--data` or default path.

Core flow:
- Parses season/week judge score columns into long format.
- Reconstructs active contestants, exit weeks, and elimination types (eliminated/withdrew).
- Maps seasons into rule regimes (`rank_classic`, `pct`, `bottom2_save`).
- Builds week-specific likelihood constraints for elimination/final ranking behavior.
- Runs dynamic blocked Metropolis-Hastings sampling per season.
- Converts latent utilities to fan-share probabilities (softmax with temperature).
- Computes per-contestant uncertainty metrics (`fan_p05`, `fan_p95`, `rsd_pct`, `rci90_pct`).
- Computes week-level consistency and margin diagnostics.

Outputs:
- `outputs/Q1/tables/Q1_1.csv` (contestant-week posterior summaries).
- `outputs/Q1/tables/Q1_2.csv` (week-level diagnostics).

#### `Q2.py`
Purpose:
- Method comparison and recommendation among rank, percent, and judge-save schemes.

Inputs:
- Data CSV.
- `outputs/Q1/tables/Q1_1.csv` (required).

Core flow:
- Cleans Q1 table and re-derives `elim_this_week` from `exit_type` + `exit_week`.
- Extracts weekly raw judge totals from `week*_judge*_score` fields.
- Constructs per-week scores under percent and rank judge components.
- Simulates judge-save elimination using bottom-two by total and judge tie-break.
- Computes agreement metrics (Kendall, Spearman, top-k Jaccard).
- Estimates fan-alignment and Monte Carlo sensitivity to fan-share perturbation.
- Aggregates overall statistics and performs MCDA with AHP-derived criteria weights.

Outputs:
- `outputs/Q2/tables/weekly_comparison_metrics.csv`
- `outputs/Q2/tables/weekly_elimination_accuracy.csv`
- `outputs/Q2/tables/summary_overall_metrics.csv`
- `outputs/Q2/tables/mcda_ahp_method_selection.csv`

#### `Q3.py`
Purpose:
- Attribute impact modeling for fan and judge channels.

Inputs:
- Data CSV.
- `outputs/Q1/tables/Q1_1.csv` (required).

Core flow:
- Converts wide judge-score data to weekly long format.
- Creates engineered features: `judge_pct`, `active_count`, lag features, cumulative fan history.
- Builds preprocessing + RandomForest pipelines (numeric imputation + one-hot encoding).
- Runs GroupKFold cross-validation grouped by season.
- Reports MAE/R2 metrics for fan model and judge model.
- Computes permutation importance tables.
- Calls `visual3.generate_q3_shap_figures(...)` for SHAP plots.

Outputs:
- `outputs/Q3/tables/metrics_cv.csv`
- `outputs/Q3/tables/permutation_importance_fans.csv`
- `outputs/Q3/tables/permutation_importance_judges.csv`
- SHAP figures in `outputs/Q3/figs/`.

#### `Q4.py`
Purpose:
- Proposes and evaluates a dynamic judge-weighting system with optional bottom-two save.

Inputs:
- Data CSV.
- `outputs/Q1/tables/Q1_1.csv` (required).

Core flow:
- Rebuilds weekly `judge_total` from raw official scores.
- Uses Q1 fan posterior summaries (`fan_pct_mean`, CI-derived `fan_share_sd`).
- Computes weekly shares `J_share`, `F_share`.
- Computes adaptive weight `w_t` as a function of uncertainty `U_t`.
- Forms combined score `C_score = w_t*J_share + (1-w_t)*F_share`.
- Applies elimination logic with optional bottom-two judges-save condition.
- Reconstructs implied season placements and computes fairness proxy (Spearman-based).
- Saves run configuration JSON for auditability.

Outputs:
- `outputs/Q4/tables/week_table_dynamic.csv`
- `outputs/Q4/tables/elimination_log_dynamic.csv`
- `outputs/Q4/tables/predicted_placements_dynamic.csv`
- `outputs/Q4/tables/metrics_dynamic.csv`
- `outputs/Q4/tables/run_config.json`

#### `visual1.py`
Purpose:
- Generate Q1 figures (Figures 5-13 style set).

Inputs:
- `outputs/Q1/tables/Q1_1.csv`, `outputs/Q1/tables/Q1_2.csv` (or explicit `--q1_1/--q1_2`).

Core flow:
- Validates required columns.
- Builds rule-level and week-level trend plots.
- Builds uncertainty distributions and uncertainty-vs-share scatter.
- Builds representative-season fan-share trajectories.
- Saves standardized filenames in Q1 fig directory.

Outputs:
- PNG files under `outputs/Q1/figs/`.

#### `visual2.py`
Purpose:
- Generate 11 publication figures for Q2 comparisons and MCDA summaries.

Inputs:
- Q2 table outputs in `outputs/Q2/tables/`.

Core flow:
- Loads weekly, elimination, summary, and MCDA tables.
- Produces season-level agreement/stability/fan-alignment plots.
- Produces elimination accuracy charts.
- Produces AHP/MCDA bars, radar, and heatmap.
- Rebuilds MCDA plotting table if CSV has incomplete scaled columns.

Outputs:
- `Fig01`-`Fig11` PNG files in `outputs/Q2/figs/`.

#### `visual3.py`
Purpose:
- Shared SHAP plotting utilities for Q3 models.

Inputs:
- Trained pipelines and feature matrices passed from `Q3.py`.

Core flow:
- Extracts encoded feature names from preprocessing pipeline.
- Computes SHAP values (TreeExplainer) on sampled encoded data.
- Generates beeswarm, bar, and top-feature dependence plots.
- Writes figures with stable prefixes (`fans_...`, `judges_...`).

Outputs:
- SHAP PNG files in `outputs/Q3/figs/`.

#### `visual4.py`
Purpose:
- Generate high-quality figures for Q4 dynamic rule behavior and impact narratives.

Inputs:
- Data CSV.
- Q4 tables: `week_table_dynamic.csv`, `metrics_dynamic.csv`, `predicted_placements_dynamic.csv`.

Core flow:
- Loads Q4 outputs and official data.
- Plots adaptive weight mechanism and uncertainty distribution.
- Plots fairness-excitement tradeoff scatter.
- Plots suspense/safety-gap distribution.
- Plots example impact spectrum across selected contestants/seasons.

Outputs:
- `Q4_fig1_...` to `Q4_fig4_...` in `outputs/Q4/figs/`.

#### `.vscode/settings.json`
Purpose:
- Local IDE environment preference (Conda env/package manager defaults).

Outputs:
- None (editor config only).

## Data Specification
Expected main data file:
- `Data/2026_MCM_Problem_C_Data.csv`

Problem statement file:
- `Data/2026_MCM_Problem_C.pdf`

Placement rules:
- Keep the CSV under `Data/` to use default behavior.
- Or override using `--data <path/to/csv>` for any script.

Important notes:
- Do not rename key columns (e.g., `season`, `celebrity_name`, `weekX_judgeY_score`).
- Q2/Q3/Q4 depend on `outputs/Q1/tables/Q1_1.csv`; run Q1 first.


## Quick Audit Checklist
1. Confirm `Data/2026_MCM_Problem_C_Data.csv` exists.
2. Run `python Q1.py` and verify `outputs/Q1/tables/Q1_1.csv` exists.
3. Run `python Q2.py`, `python Q3.py`, `python Q4.py` in order.
4. Run `python visual1.py`, `python visual2.py`, `python visual4.py` (Q3 figures are generated by `Q3.py`).
5. Inspect `outputs/Q*/tables` for CSVs and `outputs/Q*/figs` for PNGs.
