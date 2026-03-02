import os
import re
import argparse
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from project_paths import get_data_path, get_out_dir, ensure_dir


# Project-relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_Q1_OUT_DIR = get_out_dir("Q1", None) / "tables"
DEFAULT_FIG_DIR = get_out_dir("Q1", None) / "figs"

CANDIDATE_Q1_1 = [
    str(DEFAULT_Q1_OUT_DIR / "Q1_1.csv"),
]

CANDIDATE_Q1_2 = [
    str(DEFAULT_Q1_OUT_DIR / "Q1_2.csv"),
]


# Utilities
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_filename(name: str) -> str:
    """Make filename safe for Windows/macOS/Linux."""
    name = re.sub(r'[\\/*?:"<>|]', "-", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def save_fig(fig, out_dir: str, fig_id: int, title: str) -> str:
    """
    Save figure with required naming format:
      Figure x - (specific description).png
    """
    safe_mkdir(out_dir)
    fname = sanitize_filename(f"Figure {fig_id} - {title}.png")
    path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def require_columns(df: pd.DataFrame, cols, df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{df_name}] Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def resolve_path(user_path: Optional[str], candidates: List[str], desc: str) -> str:
    """
    Resolve a file path.
    - If user_path is given and exists -> use it
    - Else pick the first existing path from candidates
    """
    if user_path:
        if os.path.exists(user_path):
            return user_path
        raise FileNotFoundError(f"{desc} not found at: {user_path}")

    for p in candidates:
        if os.path.exists(p):
            return p

    msg = "\n".join(candidates)
    raise FileNotFoundError(f"Cannot find {desc}. Tried:\n{msg}")


def robust_boxplot(ax, data_by_group, labels):
    """Matplotlib boxplot wrapper that handles empty groups gracefully."""
    cleaned = []
    cleaned_labels = []
    for x, lab in zip(data_by_group, labels):
        x = pd.Series(x).dropna().values
        if len(x) > 0:
            cleaned.append(x)
            cleaned_labels.append(lab)
    if not cleaned:
        ax.text(0.5, 0.5, "No data for boxplot", ha="center", va="center")
        return
    ax.boxplot(cleaned, labels=cleaned_labels, showfliers=False)


def pick_representative_season(df1: pd.DataFrame, rule: str) -> Optional[int]:
    """
    Pick a representative season for a given rule:
    - earliest season with >= 6 weeks of data (heuristic)
    """
    if "season_rule" not in df1.columns:
        return None
    cand = df1[df1["season_rule"] == rule].copy()
    if cand.empty:
        return None
    by_season = cand.groupby("season")["week"].nunique().sort_index()
    ok = by_season[by_season >= 6]
    if len(ok) == 0:
        return int(by_season.index[0])
    return int(ok.index[0])


# Figure generators (Task 1)
def fig5_season_level_consistency(df2: pd.DataFrame, out_dir: str, fig_id: int = 5) -> str:
    """
    Figure 5: Season-level Consistency by Rule (boxplot)
    Use df2 consistency_prob; compute season-level mean over weeks with elimination events.
    """
    require_columns(df2, ["season", "season_rule", "consistency_prob", "has_elim"], "Q1_2")

    d = df2[df2["has_elim"] == 1].copy()
    season_cons = (
        d.groupby(["season_rule", "season"])["consistency_prob"]
        .mean()
        .reset_index()
        .dropna(subset=["consistency_prob"])
    )

    rules = sorted(season_cons["season_rule"].unique().tolist())
    data = [season_cons.loc[season_cons["season_rule"] == r, "consistency_prob"] for r in rules]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    robust_boxplot(ax, data, rules)
    ax.set_title("Season-level Consistency by Rule")
    ax.set_ylabel("Consistency Probability")
    ax.set_xlabel("Season Rule")
    ax.grid(True, axis="y", alpha=0.25)

    return save_fig(fig, out_dir, fig_id, "Season-level Consistency by Rule")


def fig6_week_level_uncertainty_trend(df2: pd.DataFrame, out_dir: str, fig_id: int = 6) -> str:
    """
    Figure 6: Week-level Uncertainty Trend
    Use df2 week_fan_ci90_mean: mean width of CI across contestants in that week.
    Aggregate across seasons by week: show mean and IQR band.
    """
    require_columns(df2, ["week", "week_fan_ci90_mean"], "Q1_2")
    d = df2.dropna(subset=["week_fan_ci90_mean"]).copy()

    g = d.groupby("week")["week_fan_ci90_mean"]
    stat = pd.DataFrame({
        "week": g.mean().index,
        "mean": g.mean().values,
        "q25": g.quantile(0.25).values,
        "q75": g.quantile(0.75).values,
    }).sort_values("week")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(stat["week"], stat["mean"], marker="o")
    ax.fill_between(stat["week"], stat["q25"], stat["q75"], alpha=0.2)
    ax.set_title("Week-level Uncertainty Trend (CI90 mean width)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Mean CI90 Width of Fan Share")
    ax.grid(True, alpha=0.25)

    return save_fig(fig, out_dir, fig_id, "Week-level Uncertainty Trend")


def fig7_week_level_consistency_trend(df2: pd.DataFrame, out_dir: str, fig_id: int = 7) -> str:
    """
    Figure 7: Week-level Consistency Trend
    Aggregate df2 consistency_prob across seasons by week (weeks with elimination events).
    """
    require_columns(df2, ["week", "consistency_prob", "has_elim"], "Q1_2")
    d = df2[(df2["has_elim"] == 1) & df2["consistency_prob"].notna()].copy()

    g = d.groupby("week")["consistency_prob"]
    stat = pd.DataFrame({
        "week": g.mean().index,
        "mean": g.mean().values,
        "q25": g.quantile(0.25).values,
        "q75": g.quantile(0.75).values,
    }).sort_values("week")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(stat["week"], stat["mean"], marker="o")
    ax.fill_between(stat["week"], stat["q25"], stat["q75"], alpha=0.2)
    ax.set_title("Week-level Consistency Trend")
    ax.set_xlabel("Week")
    ax.set_ylabel("Mean Consistency Probability")
    ax.grid(True, alpha=0.25)

    return save_fig(fig, out_dir, fig_id, "Week-level Consistency Trend")


def fig8_distribution_rsd(df1: pd.DataFrame, out_dir: str, fig_id: int = 8) -> str:
    """
    Figure 8: Distribution of Relative SD (RSD)
    Expect df1 rsd_pct (percentage scale).
    """
    require_columns(df1, ["rsd_pct"], "Q1_1")
    x = df1["rsd_pct"].dropna().values

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(x, bins=40)
    ax.set_title("Distribution of Relative SD (RSD)")
    ax.set_xlabel("RSD (%)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)

    return save_fig(fig, out_dir, fig_id, "Distribution of Relative SD")


def fig9_distribution_rci90(df1: pd.DataFrame, out_dir: str, fig_id: int = 9) -> str:
    """
    Figure 9: Distribution of Relative 90% CI Width
    Expect df1 rci90_pct (percentage scale).
    """
    require_columns(df1, ["rci90_pct"], "Q1_1")
    x = df1["rci90_pct"].dropna().values

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(x, bins=40)
    ax.set_title("Distribution of Relative 90% CI Width")
    ax.set_xlabel("Relative CI90 Width (%)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)

    return save_fig(fig, out_dir, fig_id, "Distribution of Relative 90% CI Width")


def fig10_uncertainty_vs_fan_share(df1: pd.DataFrame, out_dir: str, fig_id: int = 10) -> str:
    """
    Figure 10: Uncertainty vs Fan Share
    Scatter: fan_pct_mean vs rci90_pct
    """
    require_columns(df1, ["fan_pct_mean", "rci90_pct"], "Q1_1")
    d = df1.dropna(subset=["fan_pct_mean", "rci90_pct"]).copy()

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.scatter(d["fan_pct_mean"], d["rci90_pct"], s=10, alpha=0.35)
    ax.set_title("Uncertainty vs Fan Share")
    ax.set_xlabel("Posterior Mean Fan Share")
    ax.set_ylabel("Relative CI90 Width (%)")
    ax.grid(True, alpha=0.25)

    return save_fig(fig, out_dir, fig_id, "Uncertainty vs Fan Share")


def fig11_13_road_to_victory(
    df1: pd.DataFrame,
    out_dir: str,
    start_fig_id: int = 11,
    fixed_seasons: Optional[Dict] = None,
) -> List[str]:
    """
    Figures 11-13: Fan Share Trajectories for representative seasons
    - Choose a season for each rule (rank_classic, pct, bottom2_save)
    - In each season: pick top-5 contestants by fan_pct_mean in the last week, plot their trajectories
    If fixed_seasons provided, it should map rule->season, e.g. {"pct": 3, "rank_classic": 1, "bottom2_save": 28}
    """
    require_columns(df1, ["season", "week", "season_rule", "celebrity_name", "fan_pct_mean"], "Q1_1")

    rules = ["rank_classic", "pct", "bottom2_save"]
    chosen = {}

    if fixed_seasons:
        for r in rules:
            if r in fixed_seasons and fixed_seasons[r] is not None:
                chosen[r] = int(fixed_seasons[r])

    for r in rules:
        if r not in chosen:
            s = pick_representative_season(df1, r)
            if s is not None:
                chosen[r] = s

    outputs = []
    fig_id = start_fig_id

    for rule in rules:
        if rule not in chosen:
            continue
        season = chosen[rule]
        d = df1[(df1["season"] == season) & (df1["season_rule"] == rule)].copy()
        if d.empty:
            continue

        last_week = int(d["week"].max())
        last = d[d["week"] == last_week].dropna(subset=["fan_pct_mean"]).copy()
        if last.empty:
            continue

        top5 = (
            last.sort_values("fan_pct_mean", ascending=False)
            .head(5)["celebrity_name"]
            .tolist()
        )

        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        for name in top5:
            dd = d[d["celebrity_name"] == name].dropna(subset=["fan_pct_mean"]).sort_values("week")
            ax.plot(dd["week"], dd["fan_pct_mean"], marker="o", linewidth=1.5, label=name)

        ax.set_title(f"Fan Share Trajectories (Rule: {rule}, Season: {season})")
        ax.set_xlabel("Week")
        ax.set_ylabel("Posterior Mean Fan Share")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8, frameon=False)

        outputs.append(
            save_fig(fig, out_dir, fig_id, f"Fan Share Trajectories (Rule {rule}, Season {season})")
        )
        fig_id += 1

    return outputs


# Main
def main():
    ap = argparse.ArgumentParser(description="Task 1 visualization (relative-path version)")
    ap.add_argument("--data", type=str, default=None, help="Path to data csv")
    ap.add_argument("--out", type=str, default=None, help="Output directory")
    ap.add_argument("--q1_1", type=str, default=None, help="Path to Q1_1.csv (optional)")
    ap.add_argument("--q1_2", type=str, default=None, help="Path to Q1_2.csv (optional)")
    ap.add_argument("--out_dir", type=str, default=None, help="Output dir for figures (legacy)")
    # Optional: fix seasons used for trajectory figures
    ap.add_argument("--season_rank", type=int, default=None, help="Fixed season for rank_classic trajectory")
    ap.add_argument("--season_pct", type=int, default=None, help="Fixed season for pct trajectory")
    ap.add_argument("--season_save", type=int, default=None, help="Fixed season for bottom2_save trajectory")
    args = ap.parse_args()

    data_path = get_data_path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found: {data_path}")

    q1_1_path = resolve_path(args.q1_1, CANDIDATE_Q1_1, "Q1_1.csv")
    q1_2_path = resolve_path(args.q1_2, CANDIDATE_Q1_2, "Q1_2.csv")

    out_root = get_out_dir("Q1", args.out)
    out_dir = args.out_dir if args.out_dir else str(ensure_dir(out_root / "figs"))
    safe_mkdir(out_dir)

    df1 = pd.read_csv(q1_1_path)
    df2 = pd.read_csv(q1_2_path)

    print("BASE_DIR =", BASE_DIR)
    print("Q1_1 =", q1_1_path)
    print("Q1_2 =", q1_2_path)
    print("OUT_DIR =", out_dir)
    print(f"Loaded Q1_1: {df1.shape}")
    print(f"Loaded Q1_2: {df2.shape}")

    fixed = {
        "rank_classic": args.season_rank,
        "pct": args.season_pct,
        "bottom2_save": args.season_save,
    }
    # remove None entries to avoid overwriting auto-pick
    fixed = {k: v for k, v in fixed.items() if v is not None}
    fixed = fixed if fixed else None

    generated = []
    generated.append(fig5_season_level_consistency(df2, out_dir, fig_id=5))
    generated.append(fig6_week_level_uncertainty_trend(df2, out_dir, fig_id=6))
    generated.append(fig7_week_level_consistency_trend(df2, out_dir, fig_id=7))
    generated.append(fig8_distribution_rsd(df1, out_dir, fig_id=8))
    generated.append(fig9_distribution_rci90(df1, out_dir, fig_id=9))
    generated.append(fig10_uncertainty_vs_fan_share(df1, out_dir, fig_id=10))
    generated.extend(fig11_13_road_to_victory(df1, out_dir, start_fig_id=11, fixed_seasons=fixed))

    print("\nGenerated figures:")
    for p in generated:
        print(" -", p)


if __name__ == "__main__":
    main()

