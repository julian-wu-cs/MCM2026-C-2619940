# -*- coding: utf-8 -*-
"""
Q2_make_figures.py
Generate paper figures for Q2 from Q2.py CSV outputs.

Inputs:
  - outputs/Q2/tables/weekly_comparison_metrics.csv
  - outputs/Q2/tables/weekly_elimination_accuracy.csv
  - outputs/Q2/tables/summary_overall_metrics.csv
  - outputs/Q2/tables/mcda_ahp_method_selection.csv

Outputs (auto-created under outputs/Q2/figs):
  Fig01_season_rank_vs_percent_corr.png
  Fig02_season_topk_jaccard.png
  Fig03_season_fan_alignment.png
  Fig04_season_stability_band.png
  Fig05_elimination_accuracy_by_season.png
  Fig06_elimination_accuracy_overall_bar.png
  Fig07_overall_metrics_bars.png
  Fig08_ahp_weights_bar.png
  Fig09_mcda_scores_bar.png
  Fig10_mcda_radar_scaled_criteria.png
  Fig11_mcda_scaled_criteria_heatmap.png
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from project_paths import get_data_path, get_out_dir, ensure_dir

warnings.filterwarnings("ignore")

DEFAULT_Q2_TABLES_DIR = get_out_dir("Q2", None) / "tables"
DEFAULT_OUT_DIR = get_out_dir("Q2", None) / "figs"

# AHP 
AHP_WEIGHTS = {
    "fan_alignment": 0.2772,
    "stability": 0.1601,
    "elim_accuracy": 0.4673,
    "transparency": 0.0954,
}
AHP_CR = 0.0115
TOP2_METHODS = ["percent", "rank"]


def save_fig(fig, filename: str, dpi: int = 300):
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def set_plot_style():
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "DejaVu Sans"  #  SimHei

def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            f"Please verify Q2 tables were generated under outputs/Q2/tables."
        )
    return pd.read_csv(path)

def season_week_sort(df: pd.DataFrame) -> pd.DataFrame:
    if "season" in df.columns and "week" in df.columns:
        return df.sort_values(["season", "week"]).reset_index(drop=True)
    return df

def sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """ inf/-inf  NaN"""
    return df.replace([np.inf, -np.inf], np.nan)

def nanmax_or_default(a, default=1.0) -> float:
    a = np.asarray(a, dtype=float)
    if np.all(~np.isfinite(a)):
        return float(default)
    return float(np.nanmax(a))

def safe_text(ax, x, y, s, **kwargs):
    """ posx/posy should be finite values"""
    if x is None or y is None:
        return
    if not (np.isfinite(x) and np.isfinite(y)):
        return
    ax.text(x, y, s, **kwargs)

def coerce_hit_to_float(series: pd.Series) -> pd.Series:
    """
     hit_*  {0,1,NaN}
    :
      - bool True/False
      -  1/0
      -  "1"/"0"/"True"/"False"
    """
    s = series.copy()

    if s.dtype == object:
        s2 = s.astype(str).str.strip().str.lower()
        s2 = s2.replace({"true": "1", "false": "0"})
        s = pd.to_numeric(s2, errors="coerce")
    else:
        # bool -> int
        if s.dtype == bool:
            s = s.astype(float)
        else:
            s = pd.to_numeric(s, errors="coerce")

    #  0/1 NaN
    s = s.where(s.isin([0, 1]) | s.isna(), np.nan)
    return s.astype(float)

def minmax_scale_vec(x: np.ndarray, higher_better=True) -> np.ndarray:
    """
     Q2.py  min-max  MCDA 
     NaN 0
    """
    x = np.array(x, dtype=float)
    if np.all(~np.isfinite(x)):
        return np.full_like(x, np.nan)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if np.isclose(xmax - xmin, 0):
        s = np.zeros_like(x)
    else:
        s = (x - xmin) / (xmax - xmin)
    if not higher_better:
        s = 1 - s
    return s

def rebuild_mcda_for_plot(mcda: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """
     mcda CSV  scaled/score  NaN elim_accuracy 
     summary_overall_metrics.csv  3  scaled  score
     CSV DataFrame
    """
    methods = ["rank", "percent", "judge-save"]

    row = summary.iloc[0]

    #  summary  scale
    fan_align = np.array([
        row.get("fan_align_rank_mean", np.nan),
        row.get("fan_align_percent_mean", np.nan),
        row.get("fan_align_percent_mean", np.nan),  # judge-save  Q2.py  percent  total 
    ], dtype=float)

    stability = np.array([
        -row.get("sens_rank_mean", np.nan),
        -row.get("sens_percent_mean", np.nan),
        -row.get("sens_percent_mean", np.nan),
    ], dtype=float)

    elim_acc = np.array([
        row.get("elim_acc_rank", np.nan),
        row.get("elim_acc_percent", np.nan),
        row.get("elim_acc_judgesave", np.nan),
    ], dtype=float)

    transparency = np.array([1.0, 1.0, 0.7], dtype=float)

    # scale  [0,1]
    fan_s = minmax_scale_vec(fan_align, higher_better=True)
    stb_s = minmax_scale_vec(stability, higher_better=True)
    ea_s  = minmax_scale_vec(elim_acc, higher_better=True)
    tr_s  = minmax_scale_vec(transparency, higher_better=True)

    #  NaNminmax  NaN NaN NaN
    crit_scaled = np.vstack([fan_s, stb_s, ea_s, tr_s]).T  # (3,4)

    #  AHP_WEIGHTS  score Q2.py  mcda_scores = crit_scaled @ w_criteria 
    w = np.array([
        AHP_WEIGHTS["fan_alignment"],
        AHP_WEIGHTS["stability"],
        AHP_WEIGHTS["elim_accuracy"],
        AHP_WEIGHTS["transparency"],
    ], dtype=float)
    score = crit_scaled @ w

    out = pd.DataFrame({
        "method": methods,
        "score": score,
        "fan_alignment_scaled": crit_scaled[:, 0],
        "stability_scaled": crit_scaled[:, 1],
        "elim_accuracy_scaled": crit_scaled[:, 2],
        "transparency_scaled": crit_scaled[:, 3],
    })
    return out


# 2) Season 
def fig01_season_rank_vs_percent_corr(weekly: pd.DataFrame, out_dir: str):
    required = ["season", "rp_kendall", "rp_spearman"]
    for c in required:
        if c not in weekly.columns:
            raise ValueError(f"Missing column '{c}' in weekly_comparison_metrics.csv.")

    season_mean = weekly.groupby("season")[["rp_kendall", "rp_spearman"]].mean()

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(season_mean.index, season_mean["rp_kendall"], marker="o", label="Kendall (rank vs percent)")
    ax.plot(season_mean.index, season_mean["rp_spearman"], marker="s", label="Spearman (rank vs percent)")

    ax.set_title("Season-level Agreement: Rank vs Percent")
    ax.set_xlabel("Season")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1.05, 1.05)
    ax.legend()

    save_fig(fig, os.path.join(out_dir, "Fig01_season_rank_vs_percent_corr.png"))

def fig02_season_topk_jaccard(weekly: pd.DataFrame, out_dir: str):
    required = ["season", "rp_top3_jaccard", "rp_top5_jaccard", "rp_top10_jaccard"]
    for c in required:
        if c not in weekly.columns:
            raise ValueError(f"Missing column '{c}' in weekly_comparison_metrics.csv.")

    season_mean = weekly.groupby("season")[["rp_top3_jaccard", "rp_top5_jaccard", "rp_top10_jaccard"]].mean()

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(season_mean.index, season_mean["rp_top3_jaccard"], marker="o", label="Top-3 Jaccard")
    ax.plot(season_mean.index, season_mean["rp_top5_jaccard"], marker="s", label="Top-5 Jaccard")
    ax.plot(season_mean.index, season_mean["rp_top10_jaccard"], marker="^", label="Top-10 Jaccard")

    ax.set_title("Season-level Top-k Agreement (Rank vs Percent)")
    ax.set_xlabel("Season")
    ax.set_ylabel("Jaccard Similarity")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()

    save_fig(fig, os.path.join(out_dir, "Fig02_season_topk_jaccard.png"))

def fig03_season_fan_alignment(weekly: pd.DataFrame, out_dir: str):
    required = ["season", "spearman_fan_total_rank", "spearman_fan_total_percent"]
    for c in required:
        if c not in weekly.columns:
            raise ValueError(f"Missing column '{c}' in weekly_comparison_metrics.csv.")

    season_mean = weekly.groupby("season")[["spearman_fan_total_rank", "spearman_fan_total_percent"]].mean()

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(season_mean.index, season_mean["spearman_fan_total_rank"], marker="o", label="Final=Rank")
    ax.plot(season_mean.index, season_mean["spearman_fan_total_percent"], marker="s", label="Final=Percent")

    ax.set_title("Season-level Fan Alignment")
    ax.set_xlabel("Season")
    ax.set_ylabel("Spearman Correlation")
    ax.set_ylim(-1.05, 1.05)
    ax.legend()

    save_fig(fig, os.path.join(out_dir, "Fig03_season_fan_alignment.png"))

def fig04_season_stability_band(weekly: pd.DataFrame, out_dir: str):
    required = [
        "season",
        "rank_sens_mean", "rank_sens_p25", "rank_sens_p75",
        "percent_sens_mean", "percent_sens_p25", "percent_sens_p75",
    ]
    for c in required:
        if c not in weekly.columns:
            raise ValueError(f"Missing column '{c}' in weekly_comparison_metrics.csv.")

    g = weekly.groupby("season")

    rank_mean = g["rank_sens_mean"].mean()
    rank_p25  = g["rank_sens_p25"].mean()
    rank_p75  = g["rank_sens_p75"].mean()

    percent_mean = g["percent_sens_mean"].mean()
    percent_p25  = g["percent_sens_p25"].mean()
    percent_p75  = g["percent_sens_p75"].mean()

    seasons = rank_mean.index

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(seasons, rank_mean, marker="o", label="Rank sensitivity (mean)")
    ax.fill_between(seasons, rank_p25, rank_p75, alpha=0.15, label="Rank sensitivity (IQR)")

    ax.plot(seasons, percent_mean, marker="s", label="Percent sensitivity (mean)")
    ax.fill_between(seasons, percent_p25, percent_p75, alpha=0.15, label="Percent sensitivity (IQR)")

    ax.set_title("Season-level Stability Comparison")
    ax.set_xlabel("Season")
    ax.set_ylabel("Sensitivity (lower = more stable)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()

    save_fig(fig, os.path.join(out_dir, "Fig04_season_stability_band.png"))

def fig05_elimination_accuracy_by_season(elim: pd.DataFrame, out_dir: str):
    required = ["season", "hit_rank", "hit_percent", "hit_judgesave"]
    for c in required:
        if c not in elim.columns:
            raise ValueError(f"Missing column '{c}' in weekly_elimination_accuracy.csv.")

    d = elim.copy()
    d["hit_rank"]      = coerce_hit_to_float(d["hit_rank"])
    d["hit_percent"]   = coerce_hit_to_float(d["hit_percent"])
    d["hit_judgesave"] = coerce_hit_to_float(d["hit_judgesave"])

    season_stats = d.groupby("season")[["hit_rank", "hit_percent", "hit_judgesave"]].mean()

    fig = plt.figure(figsize=(12, 5))
    ax = plt.gca()

    x = np.arange(len(season_stats.index))
    w = 0.25

    ax.bar(x - w, season_stats["hit_rank"], width=w, label="Rank method")
    ax.bar(x,     season_stats["hit_percent"], width=w, label="Percent method")
    ax.bar(x + w, season_stats["hit_judgesave"], width=w, label="Judge-save method")

    ax.set_title("Elimination Prediction Accuracy by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Hit rate")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in season_stats.index])
    ax.set_ylim(0, 1.05)
    ax.legend()

    save_fig(fig, os.path.join(out_dir, "Fig05_elimination_accuracy_by_season.png"))

def fig06_elimination_accuracy_overall_bar(elim: pd.DataFrame, out_dir: str):
    required = ["hit_rank", "hit_percent", "hit_judgesave"]
    for c in required:
        if c not in elim.columns:
            raise ValueError(f"Missing column '{c}' in weekly_elimination_accuracy.csv.")

    d = elim.copy()
    d["hit_rank"]      = coerce_hit_to_float(d["hit_rank"])
    d["hit_percent"]   = coerce_hit_to_float(d["hit_percent"])
    d["hit_judgesave"] = coerce_hit_to_float(d["hit_judgesave"])

    overall = d[["hit_rank", "hit_percent", "hit_judgesave"]].mean()

    methods = ["rank", "percent", "judge-save"]
    values = [overall["hit_rank"], overall["hit_percent"], overall["hit_judgesave"]]

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.bar(methods, values)

    ax.set_title("Overall Elimination Prediction Accuracy (All Seasons)")
    ax.set_xlabel("Method")
    ax.set_ylabel("Hit rate")
    ax.set_ylim(0, 1.05)

    #  posx/posy
    for i, v in enumerate(values):
        if np.isfinite(v):
            safe_text(ax, i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")
        else:
            safe_text(ax, i, 0.02, "NaN", ha="center", va="bottom")

    save_fig(fig, os.path.join(out_dir, "Fig06_elimination_accuracy_overall_bar.png"))

def fig07_overall_metrics_bars(summary: pd.DataFrame, out_dir: str):
    row = summary.iloc[0]

    groups = {
        "Agreement (Rank vs Percent)": {
            "Kendall": row.get("rp_kendall_mean", np.nan),
            "Spearman": row.get("rp_spearman_mean", np.nan),
            "Top3": row.get("rp_top3_mean", np.nan),
            "Top5": row.get("rp_top5_mean", np.nan),
            "Top10": row.get("rp_top10_mean", np.nan),
        },
        "Fan alignment": {
            "Final=Rank": row.get("fan_align_rank_mean", np.nan),
            "Final=Percent": row.get("fan_align_percent_mean", np.nan),
        },
        "Stability (lower better)": {
            "Rank sens": row.get("sens_rank_mean", np.nan),
            "Percent sens": row.get("sens_percent_mean", np.nan),
        },
        "Elimination accuracy": {
            "Rank": row.get("elim_acc_rank", np.nan),
            "Percent": row.get("elim_acc_percent", np.nan),
            "Judge-save": row.get("elim_acc_judgesave", np.nan),
        }
    }

    labels, values, boundaries = [], [], []
    for gi, (gname, items) in enumerate(groups.items()):
        if gi > 0:
            boundaries.append(len(labels) - 0.5)
        for k, v in items.items():
            labels.append(f"{gname}\n{k}")
            values.append(v)

    fig = plt.figure(figsize=(14, 6))
    ax = plt.gca()
    x = np.arange(len(labels))
    ax.bar(x, values)

    for b in boundaries:
        ax.axvline(b, linestyle="--", linewidth=1)

    ax.set_title("Overall Metrics Summary (Q2)")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    vmax = nanmax_or_default(values, default=1.0)
    ax.set_ylim(0, max(1.05, 1.1 * vmax))

    save_fig(fig, os.path.join(out_dir, "Fig07_overall_metrics_bars.png"))

def fig08_ahp_weights_bar(weights: dict, cr: float, out_dir: str):
    keys = list(weights.keys())
    vals = np.array([weights[k] for k in keys], dtype=float)

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.bar(keys, vals)

    ax.set_title(f"AHP Criteria Weights (CR={cr:.4f})")
    ax.set_xlabel("Criteria")
    ax.set_ylabel("Weight")

    vmax = nanmax_or_default(vals, default=1.0)
    ax.set_ylim(0, vmax * 1.25 if vmax > 0 else 1.0)

    for i, v in enumerate(vals):
        safe_text(ax, i, v + 0.03 * vmax, f"{v:.4f}", ha="center", va="bottom")

    save_fig(fig, os.path.join(out_dir, "Fig08_ahp_weights_bar.png"))

def fig09_mcda_scores_bar(mcda: pd.DataFrame, out_dir: str):
    required = ["method", "score"]
    for c in required:
        if c not in mcda.columns:
            raise ValueError(f"Missing column '{c}' in mcda_ahp_method_selection.csv.")

    d = mcda.copy()
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    d = d.sort_values("score", ascending=False).reset_index(drop=True)

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.bar(d["method"], d["score"])

    ax.set_title("MCDA (AHP) Final Scores by Method")
    ax.set_xlabel("Method")
    ax.set_ylabel("Score")

    vmax = nanmax_or_default(d["score"].values, default=1.0)
    ax.set_ylim(0, max(1.05, 1.1 * vmax))

    for i, (m, s) in enumerate(zip(d["method"], d["score"])):
        if np.isfinite(s):
            safe_text(ax, i, s + 0.02 * vmax, f"{s:.3f}", ha="center", va="bottom")
            if m in TOP2_METHODS:
                safe_text(ax, i, s + 0.08 * vmax, "RECOMMEND", ha="center", va="bottom")
        else:
            safe_text(ax, i, 0.02 * vmax, "NaN", ha="center", va="bottom")

    save_fig(fig, os.path.join(out_dir, "Fig09_mcda_scores_bar.png"))

def radar_plot(ax, categories, values, label=None):
    """ NaNNaN  0 """
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()

    vals = np.array(values, dtype=float)
    angles += angles[:1]
    vals = np.append(vals, vals[0])

    ax.plot(angles, vals, marker="o", label=label)
    # fill  NaN 
    if np.all(np.isfinite(vals)):
        ax.fill(angles, vals, alpha=0.10)

def fig10_mcda_radar_scaled_criteria(mcda: pd.DataFrame, out_dir: str):
    crit_cols = ["fan_alignment_scaled", "stability_scaled", "elim_accuracy_scaled", "transparency_scaled"]
    for c in ["method"] + crit_cols:
        if c not in mcda.columns:
            raise ValueError(f"Missing column '{c}' in mcda_ahp_method_selection.csv.")

    categories = ["fan_align", "stability", "elim_acc", "transparency"]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    for _, row in mcda.iterrows():
        vals = [pd.to_numeric(row[c], errors="coerce") for c in crit_cols]
        radar_plot(ax, categories, vals, label=row["method"])

    ax.set_title("Scaled Criteria Profile by Method (Radar)")
    ax.set_xticks(np.linspace(0, 2*np.pi, len(categories), endpoint=False))
    ax.set_xticklabels(categories)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))

    save_fig(fig, os.path.join(out_dir, "Fig10_mcda_radar_scaled_criteria.png"))

def fig11_mcda_scaled_criteria_heatmap(mcda: pd.DataFrame, out_dir: str):
    crit_cols = ["fan_alignment_scaled", "stability_scaled", "elim_accuracy_scaled", "transparency_scaled"]
    for c in ["method"] + crit_cols:
        if c not in mcda.columns:
            raise ValueError(f"Missing column '{c}' in mcda_ahp_method_selection.csv.")

    d = mcda.set_index("method")[crit_cols].copy()
    data = d.apply(pd.to_numeric, errors="coerce").values.astype(float)

    fig = plt.figure(figsize=(8, 4.5))
    ax = plt.gca()

    # imshow  NaN / 0 
    im = ax.imshow(data, aspect="auto")

    ax.set_title("Scaled Criteria Heatmap by Method")
    ax.set_xlabel("Criteria")
    ax.set_ylabel("Method")

    ax.set_xticks(np.arange(len(crit_cols)))
    ax.set_xticklabels(["fan_align", "stability", "elim_acc", "transparency"])
    ax.set_yticks(np.arange(len(d.index)))
    ax.set_yticklabels(d.index.tolist())

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center")
            else:
                ax.text(j, i, "NaN", ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, os.path.join(out_dir, "Fig11_mcda_scaled_criteria_heatmap.png"))


def main(data_path, out_dir):
    set_plot_style()
    tables_dir = get_out_dir("Q2", None) / "tables"
    weekly_csv = tables_dir / "weekly_comparison_metrics.csv"
    elim_csv = tables_dir / "weekly_elimination_accuracy.csv"
    summary_csv = tables_dir / "summary_overall_metrics.csv"
    mcda_csv = tables_dir / "mcda_ahp_method_selection.csv"

    weekly  = sanitize_numeric(safe_read_csv(str(weekly_csv)))
    elim    = sanitize_numeric(safe_read_csv(str(elim_csv)))
    summary = sanitize_numeric(safe_read_csv(str(summary_csv)))
    mcda    = sanitize_numeric(safe_read_csv(str(mcda_csv)))

    weekly = season_week_sort(weekly)
    elim   = season_week_sort(elim)

    # 1)  + Top-k
    fig01_season_rank_vs_percent_corr(weekly, out_dir)
    fig02_season_topk_jaccard(weekly, out_dir)

    # 2) fan alignment
    fig03_season_fan_alignment(weekly, out_dir)

    fig04_season_stability_band(weekly, out_dir)

    # 4)  season + hit 
    fig05_elimination_accuracy_by_season(elim, out_dir)
    fig06_elimination_accuracy_overall_bar(elim, out_dir)

    # 5) Overall summary
    fig07_overall_metrics_bars(summary, out_dir)

    # 6) AHP / MCDA 
    fig08_ahp_weights_bar(AHP_WEIGHTS, AHP_CR, out_dir)

    #  mcda  scaled/score  NaN elim_accuracy_scaled  NaN summary  mcda
    crit_cols = ["fan_alignment_scaled", "stability_scaled", "elim_accuracy_scaled", "transparency_scaled"]
    need_rebuild = False
    if not set(["method", "score"] + crit_cols).issubset(mcda.columns):
        need_rebuild = True
    else:
        #  NaN  score  NaN
        tmp = mcda[["score"] + crit_cols].apply(pd.to_numeric, errors="coerce")
        if tmp.isna().all().any() or tmp["score"].isna().all():
            need_rebuild = True

    mcda_plot = rebuild_mcda_for_plot(mcda, summary) if need_rebuild else mcda

    fig09_mcda_scores_bar(mcda_plot, out_dir)
    fig10_mcda_radar_scaled_criteria(mcda_plot, out_dir)
    fig11_mcda_scaled_criteria_heatmap(mcda_plot, out_dir)

    print(f"[Done] Figures saved to: {os.path.abspath(str(out_dir))}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Q2 visualization")
    ap.add_argument("--data", type=str, default=None, help="Path to data csv")
    ap.add_argument("--out", type=str, default=None, help="Output directory")
    args = ap.parse_args()

    data_path = get_data_path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found: {data_path}")
    out_dir = ensure_dir((get_out_dir("Q2", args.out) / "figs"))

    main(data_path=data_path, out_dir=out_dir)

