# MCM 2026 Problem C - Q2
# Rank vs Percent vs Judge-Save
# Metrics: Kendall / Spearman / Top-k overlap
# Fan-bias: correlation + sensitivity (Monte Carlo perturbation of fan_share)
# Selection: MCDA + AHP

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from project_paths import get_data_path, get_out_dir, ensure_dir

# Paths
DEFAULT_Q1_PATH = get_out_dir("Q1", None) / "tables" / "Q1_1.csv"

RANDOM_STATE = 42


# Helpers: correlation & ranking
def safe_spearman(x, y):
    """
    Spearman 
     x  y  spearmanr  NaN
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(np.unique(x[~np.isnan(x)])) <= 1 or len(np.unique(y[~np.isnan(y)])) <= 1:
        return np.nan
    return spearmanr(x, y).correlation


def safe_kendall(x, y):
    """
    Kendall tau
     x  y  kendalltau  NaN
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(np.unique(x[~np.isnan(x)])) <= 1 or len(np.unique(y[~np.isnan(y)])) <= 1:
        return np.nan
    return kendalltau(x, y).correlation


def ranks_from_scores(scores, higher_is_better=True):
    """
    1 = 
    higher_is_better=True: rank=1
    """
    s = np.asarray(scores, dtype=float)
    if higher_is_better:
        order = (-s).argsort(kind="mergesort")
    else:
        order = (s).argsort(kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


def topk_overlap(rank_a, rank_b, k=3):
    """
    Top-k  Jaccard  <= k
    """
    idx_a = set(np.where(np.asarray(rank_a) <= k)[0].tolist())
    idx_b = set(np.where(np.asarray(rank_b) <= k)[0].tolist())
    if len(idx_a | idx_b) == 0:
        return np.nan
    return len(idx_a & idx_b) / len(idx_a | idx_b)


# Helpers: judge scores extraction from raw Data.csv
def infer_weeks_columns(df_data):
    """
     Data.csv 
     'week1_judge1_score', 'week10_judge3_score' 

    
      dict: week(int) -> list of columns
    """
    week_cols = {}
    for c in df_data.columns:
        c_low = c.lower()
        if c_low.startswith("week") and "judge" in c_low and c_low.endswith("_score"):
            w_part = c_low.split("_")[0]  # 'week10'
            w = int(w_part.replace("week", ""))
            week_cols.setdefault(w, []).append(c)
    return dict(sorted(week_cols.items(), key=lambda x: x[0]))


def compute_raw_judge_sum(df_data, week_cols):
    """
    season, week, celebrity_name, judge_raw_sum
     judge_raw_sum = 

    
    -  NaN/
    """
    rows = []
    base_cols = ["season", "celebrity_name"]
    for w, cols in week_cols.items():
        tmp = df_data[base_cols + cols].copy()
        tmp["week"] = w
        tmp["judge_raw_sum"] = tmp[cols].sum(axis=1, skipna=True)

        # judge_raw_sum  NaN 
        all_missing = tmp[cols].isna().all(axis=1)
        tmp.loc[all_missing, "judge_raw_sum"] = np.nan

        tmp = tmp.drop(columns=cols)
        rows.append(tmp)

    long_df = pd.concat(rows, ignore_index=True)
    long_df = long_df.dropna(subset=["judge_raw_sum"])
    return long_df


# Key FIX: build elim_this_week from Q1 exit_type/exit_week
def prepare_q1_for_q2(df_q1):
    """
    Q1_1.csv  season_rule (season, week, celebrity_name) 
    Q2 

     elim_this_week
      -  withdrew 
      -  exit_type == 'eliminated'  week == exit_week 

     df_q1 elim_this_week
    """
    df = df_q1.copy()

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    if "exit_week" in df.columns:
        df["exit_week"] = pd.to_numeric(df["exit_week"], errors="coerce")

    # 1)  season_rule season_rule 
    #     rule Q1 
    key_cols = ["season", "week", "celebrity_name"]
    if "season_rule" in df.columns:
        df = df.sort_values(["season_rule"])
        df = df.drop_duplicates(subset=key_cols, keep="first")

    # 2)  elim_this_week True
    #    withdrew
    if ("exit_type" in df.columns) and ("exit_week" in df.columns):
        is_elim = df["exit_type"].astype(str).str.lower().eq("eliminated")
        df["elim_this_week"] = (is_elim) & (df["week"] == df["exit_week"])
    else:
        df["elim_this_week"] = False

    return df


def build_weekly_frame(q1_df, judge_raw_long):
    """
     Q1  judge_pct/fan_pct_mean  raw judge sum 
    rank  judge 
    """
    df = q1_df.merge(
        judge_raw_long,
        on=["season", "week", "celebrity_name"],
        how="left"
    )
    return df


def compute_rank_based_judge_component(df_week):
    """
    rank-based 
    -  judge_raw_sum 
    -  N  N-1 ...
    -  judge_rank_pct
    """
    n = df_week.shape[0]
    order = (-df_week["judge_raw_sum"].values).argsort(kind="mergesort")
    rank_points = np.empty(n, dtype=float)
    rank_points[order] = np.arange(n, 0, -1)  # best -> n
    judge_rank_pct = rank_points / rank_points.sum()
    return pd.Series(judge_rank_pct, index=df_week.index)


def compute_total_score(judge_component, fan_component, alpha=0.5):
    """
    Total score = alpha * judge + (1-alpha) * fan
    """
    return alpha * judge_component + (1 - alpha) * fan_component


def judge_save_elimination(df_week, total_score_col, judge_metric_col="judge_pct"):
    """
    Judge-save 
    1) total_score_col  bottom2
    2) bottom2  judge_metric_col 
    3) 
    """
    df_sorted = df_week.sort_values(total_score_col, ascending=True).copy()
    if df_sorted.shape[0] < 2:
        return None
    bottom2 = df_sorted.head(2).copy()
    keep_idx = bottom2[judge_metric_col].idxmax()
    elim_idx = [i for i in bottom2.index if i != keep_idx][0]
    return df_week.loc[elim_idx, "celebrity_name"]


def run_week_metrics(method_a_rank, method_b_rank, k_list=(3, 5, 10)):
    """
    
    - Kendall tau
    - Spearman
    - Top-k Jaccard
    """
    kt = safe_kendall(method_a_rank, method_b_rank)
    sp = safe_spearman(method_a_rank, method_b_rank)

    topk = {}
    for k in k_list:
        topk[f"top{k}_jaccard"] = topk_overlap(method_a_rank, method_b_rank, k=k)

    return {"kendall": kt, "spearman": sp, **topk}


def simulate_fan_share(df_week, n_sims=2000, clip=True, seed=42):
    """
     (fan_pct_mean, fan_p05, fan_p95)  Monte Carlo 
    -  5%-95%  sigma
    -  clip  [0,1]
    -  fan share  1
    """
    rng = np.random.default_rng(seed)
    mu = df_week["fan_pct_mean"].values.astype(float)
    fan_ci90_width = df_week["fan_p95"].values.astype(float) - df_week["fan_p05"].values.astype(float)

    sigma = fan_ci90_width / (2 * 1.645)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 0.0)

    sims = rng.normal(loc=mu, scale=sigma, size=(n_sims, len(mu)))
    if clip:
        sims = np.clip(sims, 0.0, 1.0)

    row_sums = sims.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    sims = sims / row_sums
    return sims


def sensitivity_analysis_week(df_week, judge_component, alpha=0.5, n_sims=1000, seed=42):
    """
     / 
    - baseline  fan_pct_mean
    - Monte Carlo  fan share
    - sensitivity = mean(1 - Kendall_tau(baseline_rank, sim_rank))
       fan 
    """
    baseline_total = compute_total_score(judge_component, df_week["fan_pct_mean"].values, alpha=alpha)
    baseline_rank = ranks_from_scores(baseline_total, higher_is_better=True)

    sims = simulate_fan_share(df_week, n_sims=n_sims, seed=seed)
    distances = []
    for i in range(sims.shape[0]):
        total_i = compute_total_score(judge_component, sims[i, :], alpha=alpha)
        rank_i = ranks_from_scores(total_i, higher_is_better=True)
        kt = safe_kendall(baseline_rank, rank_i)
        if np.isnan(kt):
            continue
        distances.append(1 - kt)

    if len(distances) == 0:
        return {"sens_mean": np.nan, "sens_p25": np.nan, "sens_p75": np.nan}

    distances = np.array(distances, dtype=float)
    return {
        "sens_mean": float(np.mean(distances)),
        "sens_p25": float(np.quantile(distances, 0.25)),
        "sens_p75": float(np.quantile(distances, 0.75)),
    }


# AHP / MCDA
def ahp_weights(pairwise_matrix):
    """
    AHP 
    
      - w: sum=1
      - CR: 
    """
    A = np.array(pairwise_matrix, dtype=float)
    eigvals, eigvecs = np.linalg.eig(A)
    max_idx = np.argmax(eigvals.real)
    w = np.abs(eigvecs[:, max_idx].real)
    w = w / w.sum()

    n = A.shape[0]
    lambda_max = eigvals[max_idx].real
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0

    RI_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_table.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0
    return w, CR


def minmax_scale(x, higher_better=True):
    """
    Min-Max  [0,1]
     NaN NaN
    """
    x = np.array(x, dtype=float)
    if np.all(~np.isfinite(x)):
        return np.full_like(x, np.nan)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if np.isclose(xmax - xmin, 0):
        return np.zeros_like(x)
    s = (x - xmin) / (xmax - xmin)
    if not higher_better:
        s = 1 - s
    return s


# Main pipeline
def main(data_path, q1_path, out_dir, alpha=0.5, topk_list=(3, 5, 10), n_sims=1500, seed=42):
    # 1) Load
    df_data = pd.read_csv(str(data_path))
    df_q1_raw = pd.read_csv(str(q1_path))

    # 2) Q1 -> Q2 +  elim_this_week
    df_q1 = prepare_q1_for_q2(df_q1_raw)

    # 3) Build raw judge sums from Data.csv (for rank-based judge component)
    week_cols = infer_weeks_columns(df_data)
    judge_raw_long = compute_raw_judge_sum(df_data, week_cols)

    # 4) Merge into one long table
    df = build_weekly_frame(df_q1, judge_raw_long)

    required_cols = {"season", "week", "celebrity_name", "judge_pct", "fan_pct_mean", "fan_p05", "fan_p95", "elim_this_week"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Q2 is missing required columns: {missing}. Please check Q1_1.csv and merge logic.")

    # Drop rows with missing essentials ()
    df = df.dropna(subset=["judge_pct", "fan_pct_mean"])
    df = df.sort_values(["season", "week", "celebrity_name"]).reset_index(drop=True)

    weekly_records = []
    elimination_records = []

    # Loop each season-week
    for (season, week), df_week in df.groupby(["season", "week"], sort=True):
        df_week = df_week.copy().reset_index(drop=True)
        n = df_week.shape[0]
        if n < 3:
            continue

        # --- judge components ---
        judge_percent = df_week["judge_pct"].values.astype(float)
        judge_rank_pct = compute_rank_based_judge_component(df_week).values.astype(float)

        # --- fan component ---
        fan_mean = df_week["fan_pct_mean"].values.astype(float)

        # --- total scores under each method ---
        total_percent = compute_total_score(judge_percent, fan_mean, alpha=alpha)
        total_rank = compute_total_score(judge_rank_pct, fan_mean, alpha=alpha)

        # ranks (1 is best)
        rank_percent = ranks_from_scores(total_percent, higher_is_better=True)
        rank_rank = ranks_from_scores(total_rank, higher_is_better=True)

        # judge-save elimination: bottom2 by percent total, save by judge_pct
        df_week["total_percent"] = total_percent
        df_week["total_rank"] = total_rank
        elim_js = judge_save_elimination(df_week, total_score_col="total_percent", judge_metric_col="judge_pct")

        # --- (A) rank vs percent ---
        diff_rp = run_week_metrics(rank_rank, rank_percent, k_list=topk_list)

        # --- (B) rank vs judge-save; percent vs judge-save ---
        #  judge-save  percent 
        rank_judgesave = rank_percent.copy()
        diff_r_js = run_week_metrics(rank_rank, rank_judgesave, k_list=topk_list)
        diff_p_js = run_week_metrics(rank_percent, rank_judgesave, k_list=topk_list)

        # --- (C) fan alignment: Spearman(fan_share, total_score) ---
        corr_rank_fan_score = safe_spearman(fan_mean, total_rank)
        corr_percent_fan_score = safe_spearman(fan_mean, total_percent)

        # --- (D) stability via sensitivity ---
        sens_rank = sensitivity_analysis_week(df_week, judge_component=judge_rank_pct, alpha=alpha, n_sims=n_sims, seed=seed)
        sens_percent = sensitivity_analysis_week(df_week, judge_component=judge_percent, alpha=alpha, n_sims=n_sims, seed=seed)

        weekly_records.append({
            "season": season,
            "week": week,
            "n_couples": n,

            "rp_kendall": diff_rp["kendall"],
            "rp_spearman": diff_rp["spearman"],
            **{f"rp_{k}": v for k, v in diff_rp.items() if k.startswith("top")},

            "r_js_kendall": diff_r_js["kendall"],
            "r_js_spearman": diff_r_js["spearman"],
            **{f"rjs_{k}": v for k, v in diff_r_js.items() if k.startswith("top")},

            "p_js_kendall": diff_p_js["kendall"],
            "p_js_spearman": diff_p_js["spearman"],
            **{f"pjs_{k}": v for k, v in diff_p_js.items() if k.startswith("top")},

            "spearman_fan_total_rank": corr_rank_fan_score,
            "spearman_fan_total_percent": corr_percent_fan_score,

            **{f"rank_{k}": v for k, v in sens_rank.items()},
            **{f"percent_{k}": v for k, v in sens_percent.items()},
        })

        # --- elimination accuracy vs actual elim_this_week ---
        # elim_this_week == True  prepare_q1_for_q2 
        elim_actual = None
        elim_rows = df_week[df_week["elim_this_week"] == True]
        if elim_rows.shape[0] >= 1:
            elim_actual = elim_rows.iloc[0]["celebrity_name"]

        elim_rank = df_week.iloc[int(np.argmin(total_rank))]["celebrity_name"]
        elim_percent = df_week.iloc[int(np.argmin(total_percent))]["celebrity_name"]

        elimination_records.append({
            "season": season,
            "week": week,
            "actual_elim": elim_actual,
            "pred_elim_rank": elim_rank,
            "pred_elim_percent": elim_percent,
            "pred_elim_judgesave": elim_js,
            "hit_rank": (elim_actual == elim_rank) if elim_actual is not None else np.nan,
            "hit_percent": (elim_actual == elim_percent) if elim_actual is not None else np.nan,
            "hit_judgesave": (elim_actual == elim_js) if elim_actual is not None else np.nan,
        })

    weekly_df = pd.DataFrame(weekly_records)
    elim_df = pd.DataFrame(elimination_records)

    # Aggregate summaries
    summary = {}

    summary["rp_kendall_mean"] = weekly_df["rp_kendall"].mean()
    summary["rp_spearman_mean"] = weekly_df["rp_spearman"].mean()

    for k in topk_list:
        col = f"rp_top{k}_jaccard"
        if col in weekly_df.columns:
            summary[f"rp_top{k}_mean"] = weekly_df[col].mean()

    summary["fan_align_rank_mean"] = weekly_df["spearman_fan_total_rank"].mean()
    summary["fan_align_percent_mean"] = weekly_df["spearman_fan_total_percent"].mean()

    summary["sens_rank_mean"] = weekly_df["rank_sens_mean"].mean()
    summary["sens_percent_mean"] = weekly_df["percent_sens_mean"].mean()

    # elimination accuracy
    summary["elim_acc_rank"] = elim_df["hit_rank"].mean()
    summary["elim_acc_percent"] = elim_df["hit_percent"].mean()
    summary["elim_acc_judgesave"] = elim_df["hit_judgesave"].mean()

    summary_df = pd.DataFrame([summary])

    # MCDA / AHP
    methods = ["rank", "percent", "judge-save"]

    fan_align = np.array([
        summary["fan_align_rank_mean"],
        summary["fan_align_percent_mean"],
        summary["fan_align_percent_mean"],  # judge-save  percent 
    ], dtype=float)

    stability = np.array([
        -summary["sens_rank_mean"],
        -summary["sens_percent_mean"],
        -summary["sens_percent_mean"],
    ], dtype=float)

    elim_acc = np.array([
        summary["elim_acc_rank"],
        summary["elim_acc_percent"],
        summary["elim_acc_judgesave"],
    ], dtype=float)

    transparency = np.array([1.0, 1.0, 0.7], dtype=float)

    criteria_names = ["fan_alignment", "stability", "elim_accuracy", "transparency"]

    crit_scaled = np.column_stack([
        minmax_scale(fan_align, higher_better=True),
        minmax_scale(stability, higher_better=True),
        minmax_scale(elim_acc, higher_better=True),
        minmax_scale(transparency, higher_better=True),
    ])

    # AHP pairwise matrix
    A = [
        [1,   2,   1/2, 3],
        [1/2, 1,   1/3, 2],
        [2,   3,   1,   4],
        [1/3, 1/2, 1/4, 1],
    ]
    w_criteria, CR = ahp_weights(A)

    mcda_scores = crit_scaled @ w_criteria
    mcda_df = pd.DataFrame({
        "method": methods,
        "score": mcda_scores,
        "fan_alignment_scaled": crit_scaled[:, 0],
        "stability_scaled": crit_scaled[:, 1],
        "elim_accuracy_scaled": crit_scaled[:, 2],
        "transparency_scaled": crit_scaled[:, 3],
    }).sort_values("score", ascending=False).reset_index(drop=True)

    chosen_top2 = mcda_df.head(2)["method"].tolist()

    # Save outputs
    tables_dir = ensure_dir(out_dir / "tables")
    weekly_out = tables_dir / "weekly_comparison_metrics.csv"
    elim_out = tables_dir / "weekly_elimination_accuracy.csv"
    summ_out = tables_dir / "summary_overall_metrics.csv"
    mcda_out = tables_dir / "mcda_ahp_method_selection.csv"

    weekly_df.to_csv(str(weekly_out), index=False)
    elim_df.to_csv(str(elim_out), index=False)
    summary_df.to_csv(str(summ_out), index=False)
    mcda_df.to_csv(str(mcda_out), index=False)

    print("=== Q2 Finished ===")
    print(f"[Saved] {weekly_out}")
    print(f"[Saved] {elim_out}")
    print(f"[Saved] {summ_out}")
    print(f"[Saved] {mcda_out}")
    print("\nAHP criteria weights:", dict(zip(criteria_names, w_criteria.round(4))))
    print("AHP Consistency Ratio (CR):", round(float(CR), 4))
    print("Chosen top-2 methods:", chosen_top2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Q2")
    ap.add_argument("--data", type=str, default=None, help="Path to data csv")
    ap.add_argument("--out", type=str, default=None, help="Output directory")
    args = ap.parse_args()

    data_path = get_data_path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found: {data_path}")
    q1_path = DEFAULT_Q1_PATH
    if not q1_path.exists():
        raise FileNotFoundError(f"Q1 table not found: {q1_path}")
    out_dir = ensure_dir(get_out_dir("Q2", args.out))

    main(
        data_path=data_path,
        q1_path=q1_path,
        out_dir=out_dir,
        alpha=0.5,
        topk_list=(3, 5, 10),
        n_sims=1500,
        seed=42,
    )

