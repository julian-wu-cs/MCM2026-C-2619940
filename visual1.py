import os
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from project_paths import get_out_dir, ensure_dir


# -------------------------
# Utilities
# -------------------------
def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    """Make filename safe for Windows/macOS/Linux."""
    name = re.sub(r'[\\/*?:"<>|]', "-", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _save_fig(fig, out_dir: str, fig_id: int, title: str) -> str:
    """
    Save figure with required naming format:
    "Figure x - (specific description).png"
    """
    fname = _sanitize_filename(f"Figure {fig_id} - {title}.png")
    path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _require_columns(df: pd.DataFrame, cols, df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing columns: {missing}")


def _resolve_path(base_dir: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    cand = os.path.join(base_dir, p)
    if os.path.exists(cand):
        return cand
    return p  # fallback to cwd


def _ensure_week_index(g: pd.DataFrame, week_col: str, value_col: str, weeks_all: np.ndarray) -> np.ndarray:
    """Return values aligned to weeks_all (1..34), with NaN for missing."""
    m = dict(zip(g[week_col].astype(int).tolist(), g[value_col].astype(float).tolist()))
    return np.array([m.get(int(w), np.nan) for w in weeks_all], dtype=float)


# -------------------------
# Accuracy (season-level)
# -------------------------
def compute_weekly_elim_accuracy(A: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:
    """
    Define a practical, explainable "elimination prediction accuracy".

    For each (season, week) where B.has_elim==1 and B.n_elim>=1:
      - Build a heuristic score per contestant using A.fan_pct_mean and A.judge_pct
      - Predict k eliminated contestants where k = B.n_elim (lowest scores)
      - Compare against true eliminated set derived from A.exit_type=="eliminated" & A.exit_week==week
      - Weekly accuracy = |pred ∩ true| / k  (top-k hit rate)

    Heuristic scoring differs by rule:
      - pct / bottom2_save: score = judge_pct + fan_pct_mean (lower -> worse)
      - rank_classic: rank-sum proxy:
            judge_rank: rank by judge_pct (higher better)
            fan_rank:   rank by fan_pct_mean (higher better)
            combined_rank = judge_rank + fan_rank (higher -> worse)
        then take worst combined_rank (top-k worst) as predicted eliminated.

    Returns week-level rows:
      season, week, season_rule, n_elim, weekly_acc
    """
    needA = ["season", "week", "season_rule", "celebrity_name",
             "fan_pct_mean", "judge_pct", "exit_type", "exit_week"]
    needB = ["season", "week", "season_rule", "has_elim", "n_elim"]

    _require_columns(A, needA, "A")
    _require_columns(B, needB, "B")

    A2 = A.copy()
    B2 = B.copy()

    # normalize types
    A2["season"] = A2["season"].astype(int)
    A2["week"] = A2["week"].astype(int)
    B2["season"] = B2["season"].astype(int)
    B2["week"] = B2["week"].astype(int)
    B2["n_elim"] = pd.to_numeric(B2["n_elim"], errors="coerce").fillna(0).astype(int)
    B2["has_elim"] = pd.to_numeric(B2["has_elim"], errors="coerce").fillna(0).astype(int)

    rows = []
    b_elim = B2[(B2["has_elim"] == 1) & (B2["n_elim"] > 0)].copy()
    if b_elim.empty:
        return pd.DataFrame(columns=["season", "week", "season_rule", "n_elim", "weekly_acc"])

    for (season, week), bw in b_elim.groupby(["season", "week"], sort=True):
        season = int(season)
        week = int(week)
        season_rule = str(bw["season_rule"].iloc[0])
        k = int(bw["n_elim"].iloc[0])

        a_w = A2[(A2["season"] == season) & (A2["week"] == week)].copy()
        if a_w.empty:
            rows.append({"season": season, "week": week, "season_rule": season_rule, "n_elim": k, "weekly_acc": np.nan})
            continue

        # true eliminated set
        true_elim = a_w[(a_w["exit_type"].astype(str) == "eliminated") &
                        (a_w["exit_week"].fillna(-1).astype(int) == week)]
        true_set = set(true_elim["celebrity_name"].astype(str).tolist())
        if len(true_set) == 0:
            rows.append({"season": season, "week": week, "season_rule": season_rule, "n_elim": k, "weekly_acc": np.nan})
            continue

        # build prediction candidates
        a_w["fan_pct_mean"] = pd.to_numeric(a_w["fan_pct_mean"], errors="coerce")
        a_w["judge_pct"] = pd.to_numeric(a_w["judge_pct"], errors="coerce")
        a_w = a_w.replace([np.inf, -np.inf], np.nan).dropna(subset=["fan_pct_mean", "judge_pct"])
        if a_w.empty:
            rows.append({"season": season, "week": week, "season_rule": season_rule, "n_elim": k, "weekly_acc": np.nan})
            continue

        # exclude withdrew (optional but sensible for evaluation)
        a_w = a_w[a_w["exit_type"].astype(str) != "withdrew"].copy()
        if a_w.empty:
            rows.append({"season": season, "week": week, "season_rule": season_rule, "n_elim": k, "weekly_acc": np.nan})
            continue

        # prediction
        if season_rule in ["pct", "bottom2_save"]:
            a_w["score"] = a_w["judge_pct"] + a_w["fan_pct_mean"]  # lower worse
            pred = a_w.sort_values("score", ascending=True).head(k)["celebrity_name"].astype(str).tolist()
        else:
            a_w["judge_rank"] = a_w["judge_pct"].rank(ascending=False, method="average")
            a_w["fan_rank"] = a_w["fan_pct_mean"].rank(ascending=False, method="average")
            a_w["combined_rank"] = a_w["judge_rank"] + a_w["fan_rank"]  # higher worse
            pred = a_w.sort_values("combined_rank", ascending=False).head(k)["celebrity_name"].astype(str).tolist()

        pred_set = set(pred)
        hit = len(pred_set & true_set)
        weekly_acc = hit / max(k, 1)

        rows.append({"season": season, "week": week, "season_rule": season_rule,
                     "n_elim": k, "weekly_acc": float(weekly_acc)})

    return pd.DataFrame(rows)


def compute_season_accuracy(acc_week: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate weekly accuracy to season-level accuracy by averaging over elimination weeks (within a season).
    """
    if acc_week.empty:
        return pd.DataFrame(columns=["season_rule", "season", "season_accuracy", "n_weeks_used"])

    g = (
        acc_week.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["weekly_acc"])
        .groupby(["season_rule", "season"], as_index=False)
        .agg(season_accuracy=("weekly_acc", "mean"),
             n_weeks_used=("weekly_acc", "count"))
    )
    return g


# -------------------------
# Figure functions
# -------------------------
def fig1_season_consistency_by_rule(B: pd.DataFrame, out_dir: str, fig_id: int) -> str:
    _require_columns(B, ["season", "season_rule", "consistency_prob"], "B")

    season_level = (
        B.groupby(["season_rule", "season"], as_index=False)
         .agg(season_consistency=("consistency_prob", "mean"))
    )

    rules = sorted(season_level["season_rule"].dropna().unique().tolist())
    data = [season_level.loc[season_level["season_rule"] == r, "season_consistency"].dropna().values for r in rules]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.boxplot(data, labels=rules, showmeans=True)
    ax.set_title(f"Figure {fig_id} - Season-level Consistency by Rule")
    ax.set_xlabel("Elimination rule")
    ax.set_ylabel("Mean consistency probability (per season)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    return _save_fig(fig, out_dir, fig_id, "Season-level Consistency by Rule")


def fig2_week_uncertainty_trend_fullweeks(B: pd.DataFrame, out_dir: str, fig_id: int) -> str:
    """
    Week-level uncertainty trend (mean over seasons), aligned to weeks 1..34.
    """
    _require_columns(B, ["week", "season_rule", "week_fan_ci90_mean"], "B")

    weeks_all = np.arange(1, 35)
    g = (
        B.groupby(["season_rule", "week"], as_index=False)
         .agg(mean_ci90=("week_fan_ci90_mean", "mean"))
    )

    rules = sorted(g["season_rule"].dropna().unique().tolist())
    fig, ax = plt.subplots(figsize=(9.2, 4.9))

    for r in rules:
        sub = g[g["season_rule"] == r].sort_values("week")
        y = _ensure_week_index(sub, "week", "mean_ci90", weeks_all)
        ax.plot(weeks_all, y, marker="o", linewidth=1.6, label=r)

    ax.set_title(f"Figure {fig_id} - Week-level Uncertainty Trend (Mean 90% CI Width)")
    ax.set_xlabel("Week index (1-34)")
    ax.set_ylabel("Mean 90% CI width (fan share)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Rule", ncol=3, fontsize=9, title_fontsize=10)

    return _save_fig(fig, out_dir, fig_id, "Week-level Uncertainty Trend (Mean 90% CI Width)")


def fig3_week_consistency_trend_fullweeks(B: pd.DataFrame, out_dir: str, fig_id: int) -> str:
    """
    Week-level consistency trend (mean over seasons), aligned to weeks 1..34.
    """
    _require_columns(B, ["week", "season_rule", "consistency_prob"], "B")

    weeks_all = np.arange(1, 35)
    g = (
        B.groupby(["season_rule", "week"], as_index=False)
         .agg(mean_consistency=("consistency_prob", "mean"))
    )

    rules = sorted(g["season_rule"].dropna().unique().tolist())
    fig, ax = plt.subplots(figsize=(9.2, 4.9))

    for r in rules:
        sub = g[g["season_rule"] == r].sort_values("week")
        y = _ensure_week_index(sub, "week", "mean_consistency", weeks_all)
        ax.plot(weeks_all, y, marker="o", linewidth=1.6, label=r)

    ax.set_title(f"Figure {fig_id} - Week-level Consistency Trend (Mean across Seasons)")
    ax.set_xlabel("Week index (1-34)")
    ax.set_ylabel("Mean consistency probability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Rule", ncol=3, fontsize=9, title_fontsize=10)

    return _save_fig(fig, out_dir, fig_id, "Week-level Consistency Trend (Mean across Seasons)")


def fig4_rsd_distribution(A: pd.DataFrame, out_dir: str, fig_id: int) -> str:
    _require_columns(A, ["rsd_pct"], "A")
    x = A["rsd_pct"].replace([np.inf, -np.inf], np.nan).dropna().values
    if len(x) == 0:
        raise ValueError("A.rsd_pct has no valid numeric values.")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(x, bins=40)
    ax.set_title(f"Figure {fig_id} - Distribution of Relative SD (rsd_pct)")
    ax.set_xlabel("Relative standard deviation (%)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)

    return _save_fig(fig, out_dir, fig_id, "Distribution of Relative SD (rsd_pct)")


def fig5_rci90_distribution(A: pd.DataFrame, out_dir: str, fig_id: int) -> str:
    _require_columns(A, ["rci90_pct"], "A")
    x = A["rci90_pct"].replace([np.inf, -np.inf], np.nan).dropna().values
    if len(x) == 0:
        raise ValueError("A.rci90_pct has no valid numeric values.")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(x, bins=40)
    ax.set_title(f"Figure {fig_id} - Distribution of Relative 90% CI Width (rci90_pct)")
    ax.set_xlabel("Relative 90% CI width (%)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)

    return _save_fig(fig, out_dir, fig_id, "Distribution of Relative 90% CI Width (rci90_pct)")


def fig6_uncertainty_vs_share(A: pd.DataFrame, out_dir: str, fig_id: int) -> str:
    _require_columns(A, ["fan_pct_mean", "rci90_pct", "season_rule"], "A")
    df = A[["fan_pct_mean", "rci90_pct", "season_rule"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        raise ValueError("A has no valid rows for fan_pct_mean vs rci90_pct scatter.")

    rules = sorted(df["season_rule"].dropna().unique().tolist())
    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    for r in rules:
        sub = df[df["season_rule"] == r]
        ax.scatter(sub["fan_pct_mean"].values, sub["rci90_pct"].values, s=18, alpha=0.65, label=r)

    ax.set_title(f"Figure {fig_id} - Uncertainty vs Fan Share (rci90_pct vs mean share)")
    ax.set_xlabel("Mean fan share (fan_pct_mean)")
    ax.set_ylabel("Relative 90% CI width (%) (rci90_pct)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Rule", ncol=3, fontsize=9, title_fontsize=10)

    return _save_fig(fig, out_dir, fig_id, "Uncertainty vs Fan Share (rci90_pct vs mean share)")


def fig7plus_example_season_trajectories(A: pd.DataFrame, out_dir: str, start_fig_id: int):
    """
    For each rule, pick one representative season and plot trajectories
    of top-K contestants' mean fan share over weeks, with [p05, p95] bands.
    """
    needed = ["season", "week", "season_rule", "celebrity_name", "fan_pct_mean", "fan_p05", "fan_p95"]
    _require_columns(A, needed, "A")

    df = A[needed].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return []

    paths = []
    fig_id = start_fig_id

    for rule in sorted(df["season_rule"].unique().tolist()):
        sub_rule = df[df["season_rule"] == rule]
        seasons = sorted(sub_rule["season"].unique().tolist())
        if not seasons:
            continue
        season_pick = seasons[0]
        sdata = sub_rule[sub_rule["season"] == season_pick].copy()

        topk = (
            sdata.groupby("celebrity_name", as_index=False)["fan_pct_mean"].mean()
                 .sort_values("fan_pct_mean", ascending=False)
                 .head(5)["celebrity_name"].tolist()
        )

        fig, ax = plt.subplots(figsize=(9.2, 5.2))

        for name in topk:
            tdata = sdata[sdata["celebrity_name"] == name].sort_values("week")
            w = tdata["week"].values
            mu = tdata["fan_pct_mean"].values
            p05 = tdata["fan_p05"].values
            p95 = tdata["fan_p95"].values
            ax.plot(w, mu, marker="o", linewidth=1.6, label=name)
            ax.fill_between(w, p05, p95, alpha=0.18)

        ax.set_title(f"Figure {fig_id} - Fan Share Trajectories (Rule={rule}, Season={season_pick})")
        ax.set_xlabel("Week index")
        ax.set_ylabel("Mean fan share (with 90% credible band)")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Contestant (Top 5)", fontsize=8, title_fontsize=10, loc="best")

        title = f"Fan Share Trajectories (Rule={rule}, Season={season_pick})"
        paths.append(_save_fig(fig, out_dir, fig_id, title))
        fig_id += 1

    return paths


def fig8_season_accuracy_by_rule(season_acc: pd.DataFrame, out_dir: str, fig_id: int) -> str:
    """
    Boxplot of season-level elimination prediction accuracy by rule.
    """
    _require_columns(season_acc, ["season_rule", "season_accuracy"], "season_acc")
    rules = sorted(season_acc["season_rule"].dropna().unique().tolist())
    data = [season_acc.loc[season_acc["season_rule"] == r, "season_accuracy"].dropna().values for r in rules]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.boxplot(data, labels=rules, showmeans=True)
    ax.set_title(f"Figure {fig_id} - Season-level Elimination Prediction Accuracy by Rule")
    ax.set_xlabel("Elimination rule")
    ax.set_ylabel("Mean top-k hit rate per season")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    return _save_fig(fig, out_dir, fig_id, "Season-level Elimination Prediction Accuracy by Rule")


def fig9_accuracy_vs_consistency(season_acc: pd.DataFrame, B: pd.DataFrame, out_dir: str, fig_id: int) -> str:
    """
    Scatter: season accuracy vs season consistency (mean over weeks), colored by rule.
    """
    _require_columns(season_acc, ["season_rule", "season", "season_accuracy"], "season_acc")
    _require_columns(B, ["season_rule", "season", "consistency_prob"], "B")

    season_cons = (
        B.groupby(["season_rule", "season"], as_index=False)
         .agg(season_consistency=("consistency_prob", "mean"))
    )

    df = season_acc.merge(season_cons, on=["season_rule", "season"], how="inner")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["season_accuracy", "season_consistency"])
    if df.empty:
        raise ValueError("No valid rows for accuracy vs consistency scatter.")

    rules = sorted(df["season_rule"].dropna().unique().tolist())
    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    for r in rules:
        sub = df[df["season_rule"] == r]
        ax.scatter(sub["season_consistency"].values, sub["season_accuracy"].values, s=28, alpha=0.75, label=r)

    ax.set_title(f"Figure {fig_id} - Season Accuracy vs Season Consistency")
    ax.set_xlabel("Season consistency (mean over weeks)")
    ax.set_ylabel("Season elimination prediction accuracy")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Rule", ncol=3, fontsize=9, title_fontsize=10)

    return _save_fig(fig, out_dir, fig_id, "Season Accuracy vs Season Consistency")


# -------------------------
# Main
# -------------------------
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_a = str(get_out_dir("Q1", None) / "tables" / "Q1_1.csv")
    default_b = str(get_out_dir("Q1", None) / "tables" / "Q1_2.csv")
    default_out = str(get_out_dir("Q1", None) / "figs")

    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=str, default=default_a,
                    help="A table path (contestant-week). Default: outputs/Q1/tables/Q1_1.csv")
    ap.add_argument("--b", type=str, default=default_b,
                    help="B table path (week-level). Default: outputs/Q1/tables/Q1_2.csv")
    ap.add_argument("--out", type=str, default=default_out,
                    help="Output directory for figures. Default: outputs/Q1/figs")
    args = ap.parse_args()

    a_path = _resolve_path(base_dir, args.a)
    b_path = _resolve_path(base_dir, args.b)
    out_dir = _resolve_path(base_dir, args.out)
    _safe_mkdir(out_dir)
    ensure_dir(get_out_dir("Q1", None) / "figs")

    if not os.path.exists(a_path):
        raise FileNotFoundError(f"A table not found: {a_path}")
    if not os.path.exists(b_path):
        raise FileNotFoundError(f"B table not found: {b_path}")

    A = pd.read_csv(a_path)
    B = pd.read_csv(b_path)

    # sanity checks
    _require_columns(A, ["fan_pct_mean", "fan_p05", "fan_p95", "rsd_pct", "rci90_pct"], "A")
    _require_columns(B, ["week_fan_ci90_mean", "consistency_prob", "has_elim", "n_elim"], "B")

    # ---- compute accuracy tables ----
    acc_week = compute_weekly_elim_accuracy(A, B)
    season_acc = compute_season_accuracy(acc_week)

    # save evaluation tables
    acc_week_path = os.path.join(out_dir, "season_week_accuracy.csv")
    season_acc_path = os.path.join(out_dir, "season_accuracy.csv")
    acc_week.to_csv(acc_week_path, index=False)
    season_acc.to_csv(season_acc_path, index=False)

    # ---- figures ----
    paths = []
    fig_idx = 1

    paths.append(fig1_season_consistency_by_rule(B, out_dir, fig_idx)); fig_idx += 1
    paths.append(fig2_week_uncertainty_trend_fullweeks(B, out_dir, fig_idx)); fig_idx += 1
    paths.append(fig3_week_consistency_trend_fullweeks(B, out_dir, fig_idx)); fig_idx += 1
    paths.append(fig4_rsd_distribution(A, out_dir, fig_idx)); fig_idx += 1
    paths.append(fig5_rci90_distribution(A, out_dir, fig_idx)); fig_idx += 1
    paths.append(fig6_uncertainty_vs_share(A, out_dir, fig_idx)); fig_idx += 1

    paths.extend(fig7plus_example_season_trajectories(A, out_dir, fig_idx))
    fig_idx += len(paths)  # keep ids monotonic (not strictly needed for filenames)

    if not season_acc.empty:
        paths.append(fig8_season_accuracy_by_rule(season_acc, out_dir, fig_idx)); fig_idx += 1
        paths.append(fig9_accuracy_vs_consistency(season_acc, B, out_dir, fig_idx)); fig_idx += 1

    print("Saved evaluation tables:")
    print(" -", acc_week_path)
    print(" -", season_acc_path)

    print("Saved figures:")
    for p in paths:
        print(" -", p)


if __name__ == "__main__":
    main()


