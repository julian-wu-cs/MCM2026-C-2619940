
# ===============================
# visual_q4.py
# High-Quality Paper Figures for Q4 (Publication Ready)
# Style: Academic/Science (Clean, High Contrast, Annotated)
# ===============================

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, spearmanr
import matplotlib.patheffects as path_effects
from project_paths import get_data_path, get_out_dir, ensure_dir

# -------------------------
# 1. Aesthetics Configuration
# -------------------------
# Professional Color Palette (Colorblind Friendly)
COLORS = {
    'judge': '#2C3E50',      # Dark Blue/Slate
    'fan': '#E74C3C',        # Vibrant Red
    'system': '#27AE60',     # Emerald Green
    'neutral': '#95A5A6',    # Concrete Grey
    'highlight': '#F39C12',  # Orange
    'bg_grid': '#ECF0F1',    # Very Light Grey
    'text': '#34495E'
}

plt.style.use('default') # Reset
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.color': '#B0B0B0',
    'grid.linestyle': '--',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (12, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})

# -------------------------
# 2. Path Setup
# -------------------------
ap = argparse.ArgumentParser(description="Q4 visualization")
ap.add_argument("--data", type=str, default=None, help="Path to data csv")
ap.add_argument("--out", type=str, default=None, help="Output directory")
args = ap.parse_args()

OFFICIAL_CSV = str(get_data_path(args.data))
Q4_TABLES_DIR = get_out_dir("Q4", None) / "tables"
FIG_DIR = str(ensure_dir(get_out_dir("Q4", args.out) / "figs"))

# Files
WEEK_TABLE = str(Q4_TABLES_DIR / "week_table_dynamic.csv")
METRICS = str(Q4_TABLES_DIR / "metrics_dynamic.csv")
PRED_PLAC = str(Q4_TABLES_DIR / "predicted_placements_dynamic.csv")

# -------------------------
# 3. Data Loading
# -------------------------
def load_data():
    try:
        week_df = pd.read_csv(WEEK_TABLE)
        met_df  = pd.read_csv(METRICS)
        pred_df = pd.read_csv(PRED_PLAC)
        official_df = pd.read_csv(OFFICIAL_CSV)
        return week_df, met_df, pred_df, official_df
    except Exception as e:
        print(f"Critical Error: {e}")
        return None, None, None, None

week_df, met_df, pred_df, official_df = load_data()
if week_df is None: exit(1)

# -------------------------
# 4. Plotting Functions
# -------------------------

def plot_adaptive_mechanism():
    """
    Figure 1: Dual-Axis Plot
    X-axis: Uncertainty (U_t)
    Left Y-axis: Judge Weight (w_t) - Curve
    Right Y-axis: Frequency Density (KDE) - Area
    """
    print("Generating Q4_fig1...")
    
    # Prepare data
    data = week_df[["U_t", "w_t"]].drop_duplicates().sort_values("U_t")
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Background: Uncertainty Distribution (KDE)
    ax2 = ax1.twinx()
    kde = gaussian_kde(data["U_t"])
    x_grid = np.linspace(data["U_t"].min(), data["U_t"].max(), 200)
    ax2.fill_between(x_grid, kde(x_grid), color=COLORS['neutral'], alpha=0.10)
    ax2.plot(x_grid, kde(x_grid), color=COLORS['neutral'], alpha=0.45, linewidth=1.5)
    ax2.set_ylabel('Frequency Density (Weeks)', color=COLORS['neutral'], labelpad=15)
    ax2.tick_params(axis='y', colors=COLORS['neutral'])
    ax2.grid(False) # Turn off secondary grid
    
    # Foreground: The Mechanism Curve
    # Scatter points with gradient color based on U_t
    sc = ax1.scatter(
        data["U_t"],
        data["w_t"],
        c=data["U_t"],
        cmap='Oranges',
        alpha=0.9,
        s=60,
        label='Actual Weeks',
        zorder=5,
        edgecolors='white',
        linewidths=0.4
    )
    
    # Theoretical Curve
    w_min, w_max = 0.35, 0.70
    kappa = 0.02
    y_curve = w_min + (w_max - w_min) * (x_grid / (x_grid + kappa))
    
    ax1.plot(x_grid, y_curve, color=COLORS['judge'], linewidth=4, label='Adaptive Function', zorder=4)
    
    # Annotations
    ax1.set_xlabel('Fan Vote Uncertainty ($U_t$)', fontweight='bold')
    ax1.set_ylabel('Judge Weight ($w_t$)', color=COLORS['judge'], fontweight='bold', labelpad=15)
    ax1.tick_params(axis='y', colors=COLORS['judge'])
    
    # Zone Labels with Arrows
    ax1.annotate("Lower uncertainty\n($w_t \\approx 0.35$)",
                 xy=(0.002, 0.38), xytext=(0.02, 0.45),
                 arrowprops=dict(facecolor=COLORS['fan'], shrink=0.05, alpha=0.5),
                 fontsize=11, color=COLORS['fan'], fontweight='bold')
                 
    ax1.annotate("Higher uncertainty\n($w_t \\to 0.70$)",
                 xy=(0.08, 0.68), xytext=(0.04, 0.60),
                 arrowprops=dict(facecolor=COLORS['judge'], shrink=0.05, alpha=0.5),
                 fontsize=11, color=COLORS['judge'], fontweight='bold', ha='right')

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COLORS['neutral'], alpha=0.10, edgecolor='none', label='Distribution of $U_t$'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=plt.get_cmap('Oranges')(0.6),
               markeredgecolor='white', markeredgewidth=0.6, markersize=8, label='Actual Weeks'),
        Line2D([0], [0], color=COLORS['judge'], linewidth=4, label='Adaptive Function')
    ]
    ax1.legend(handles=legend_handles, loc='lower right', frameon=True, framealpha=0.9, title="Components")
    
    plt.title('Adaptive Weighting: Judge Weight vs. Uncertainty', pad=20)
    
    path = os.path.join(FIG_DIR, "Q4_fig1_adaptive_weight.png")
    plt.savefig(path)
    plt.close()

def plot_tradeoff_quadrant():
    """
    Figure 2: Scatter Plot with Quadrants + Density
    X: Excitement (Upset Rate)
    Y: Fairness (Spearman)
    """
    print("Generating Q4_fig2...")

    def simulate_placements(season_week_df: pd.DataFrame, score_col: str) -> dict:
        active = set(season_week_df["celebrity_name"].unique())
        elimination_order = []
        for week, gw in season_week_df.groupby("week"):
            gw = gw[gw["celebrity_name"].isin(active)]
            if len(gw) <= 1:
                continue
            eliminated = gw.loc[gw[score_col].idxmin(), "celebrity_name"]
            if eliminated in active:
                active.remove(eliminated)
                elimination_order.append(eliminated)
        remaining = list(active)
        final_order = remaining + elimination_order[::-1]
        return {name: idx + 1 for idx, name in enumerate(final_order)}

    def season_skill(season_week_df: pd.DataFrame) -> pd.Series:
        skill = season_week_df.dropna(subset=["judge_total"]).groupby("celebrity_name")["judge_total"].mean()
        return skill

    def season_upset_rate(season_week_df: pd.DataFrame, score_col: str) -> float:
        g_clean = season_week_df.dropna(subset=["J_share", score_col])
        upsets = 0
        weeks = 0
        for week, gw in g_clean.groupby("week"):
            if len(gw) < 1:
                continue
            topJ = gw.loc[gw["J_share"].idxmax(), "celebrity_name"]
            topC = gw.loc[gw[score_col].idxmax(), "celebrity_name"]
            upsets += int(topJ != topC)
            weeks += 1
        return upsets / weeks if weeks > 0 else 0.0

    season_rows = []
    for season, g in week_df.groupby("season"):
        g = g.dropna(subset=["J_share", "F_share", "w_t", "C_score"])
        if len(g) == 0:
            continue
        g = g.copy()
        g["C_50"] = 0.5 * g["J_share"] + 0.5 * g["F_share"]

        skill = season_skill(g)

        place_dw = simulate_placements(g, "C_score")
        place_50 = simulate_placements(g, "C_50")

        def fairness_from_places(place_map: dict) -> float:
            common = sorted(set(place_map.keys()) & set(skill.index))
            if len(common) < 3:
                return np.nan
            placement = np.array([place_map[n] for n in common], dtype=float)
            skill_vals = skill.loc[common].to_numpy(dtype=float)
            r, _ = spearmanr(placement, skill_vals)
            return float(abs(r)) if np.isfinite(r) else np.nan

        season_rows.append({
            "season": season,
            "rule": "DW-UW",
            "upset_rate": season_upset_rate(g, "C_score"),
            "fairness": fairness_from_places(place_dw)
        })
        season_rows.append({
            "season": season,
            "rule": "Fixed 50/50",
            "upset_rate": season_upset_rate(g, "C_50"),
            "fairness": fairness_from_places(place_50)
        })

    df = pd.DataFrame(season_rows).dropna(subset=["upset_rate", "fairness"])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    base = df[df["rule"] == "Fixed 50/50"]
    dwuw = df[df["rule"] == "DW-UW"]

    sns.kdeplot(data=base, x="upset_rate", y="fairness", fill=True, cmap="Greens", alpha=0.2, ax=ax, levels=5, thresh=0.1)
    
    sns.scatterplot(
        data=base,
        x="upset_rate",
        y="fairness",
        s=120,
        marker='o',
        color=COLORS['neutral'],
        edgecolor='white',
        linewidth=1.2,
        ax=ax,
        zorder=4,
        label="Fixed 50/50"
    )
    sns.scatterplot(
        data=dwuw,
        x="upset_rate",
        y="fairness",
        s=150,
        marker='o',
        color=COLORS['judge'],
        edgecolor='white',
        linewidth=1.5,
        ax=ax,
        zorder=5,
        label="DW-UW"
    )
    
    # Label key seasons
    for _, row in dwuw.iterrows():
        if row['season'] in [1, 27, 34] or row['fairness'] < 0.93 or row['upset_rate'] < 0.4 or row['upset_rate'] > 0.8:
            ax.text(row['upset_rate']+0.01, row['fairness']+0.002, f"S{int(row['season'])}", fontsize=10, fontweight='bold', alpha=0.8)

    ax.set_xlabel('Upset Rate (Combined Winner $\\ne$ Judge Winner)', fontweight='bold')
    ax.set_ylabel('Fairness: Spearman Correlation Magnitude ($|r_s|$)', fontweight='bold')
    # No invert_yaxis needed now because 1.0 is "Good" and is at the top
    ax.set_ylim(0.85, 1.03) 
    ax.set_xlim(0.2, 1.05)
    
    plt.title('Fairness vs. Upset Rate (Season Level)', pad=20)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    path = os.path.join(FIG_DIR, "Q4_fig2_fairness_excitement.png")
    with plt.rc_context({'savefig.bbox': 'standard'}):
        fig.savefig(path)
    plt.close(fig)

def plot_suspense_violin():
    """
    Figure 3: Raincloud Plot of Score Gaps
    """
    print("Generating Q4_fig3...")
    
    deltas = []
    for (s, w), g in week_df.groupby(["season", "week"]):
        g = g.dropna(subset=["C_score"]).sort_values("C_score")
        if len(g) >= 3:
            # Gap between elimination cutoff (lowest safe) and eliminated (lowest)
            # Assuming 'results' implies ranking. C_score is the metric.
            # g is sorted by C_score ascending. g.iloc[0] is lowest (eliminated). g.iloc[1] is 2nd lowest.
            # In bottom-two, gap is between 1 and 0.
            delta = float(g.iloc[1]["C_score"] - g.iloc[0]["C_score"])
            deltas.append(delta)
    
    data = pd.DataFrame({"Gap": deltas})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Half Violin (Density)
    sns.violinplot(x=data["Gap"], color=COLORS['highlight'], alpha=0.4, inner=None, ax=ax, orient='h', cut=0)
    
    # Strip Plot (Raw Data) - Shifted down
    # Note: Seaborn's raincloud support is tricky, doing manual overlap
    sns.stripplot(x=data["Gap"], color=COLORS['judge'], alpha=0.15, size=3, jitter=True, ax=ax, orient='h')
    
    # Box Plot (Summary) - Inside violin
    sns.boxplot(x=data["Gap"], color='white', width=0.1, ax=ax, orient='h', showfliers=False, 
                boxprops={'facecolor':'none', 'edgecolor':'#333333'})
    
    # Median Line Annotation
    median = np.median(deltas)
    ax.axvline(median, color=COLORS['fan'], linestyle='--', linewidth=2)
    ax.text(median, -0.42, f"Median gap: {median:.3f}", color=COLORS['fan'], fontweight='bold', ha='center', backgroundcolor='white')
    
    # Limit X axis to show the "Tight" nature
    ax.set_xlim(0, 0.06)
    
    ax.set_xlabel('Score Gap (Lowest Safe - Eliminated)', fontweight='bold')
    ax.set_yticks([]) 
    
    ax.text(0.05, 0.3, "Smaller gaps indicate\ncloser eliminations.", 
            fontsize=12, color=COLORS['judge'], ha='center', va='center', 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round'))
    
    plt.title('Suspense Distribution: Weekly Safety Margins', pad=20)
    
    path = os.path.join(FIG_DIR, "Q4_fig3_suspense_gap.png")
    plt.savefig(path)
    plt.close()

def plot_impact_spectrum():
    """
    Figure 4: Slope Chart - Multi-Season Impact Spectrum
    Shows 5 Representative Cases:
    1. Correction (Bobby Bones)
    2. Validation (Donald Driver)
    3. Redemption (Juan Pablo)
    4. Stability (Snooki)
    5. The "Good" Upset (Iman Shumpert)
    """
    print("Generating Q4_fig4...")
    
    # Manually defined for cleanliness and diversity (based on analysis)
    # This ensures we tell the "Spectrum" story
    # Added offsets (lo=left_offset, ro=right_offset) to prevent overlaps
    cases = [
        {'name': 'Bobby Bones', 'season': 27, 'official': 1, 'dw_uw': 4, 'type': 'Decrease (high uncertainty)', 'lo': 0.1, 'ro': 0},
        {'name': 'Milo Manheim', 'season': 27, 'official': 2, 'dw_uw': 1, 'type': 'Increase (technical merit)', 'lo': 0, 'ro': 0.1},
        {'name': 'Donald Driver', 'season': 14, 'official': 1, 'dw_uw': 1, 'type': 'No change (low uncertainty)', 'lo': -0.1, 'ro': -0.1},
        {'name': 'Juan Pablo', 'season': 27, 'official': 6, 'dw_uw': 3, 'type': 'Conditional save', 'lo': 0, 'ro': 0}, 
        {'name': 'Snooki', 'season': 17, 'official': 8, 'dw_uw': 8, 'type': 'No change', 'lo': 0, 'ro': 0}
    ]
    
    df = pd.DataFrame(cases)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scales
    left_x, right_x = 0, 1
    
    # Draw Lines
    for idx, row in df.iterrows():
        name = row['name']
        start = row['official']
        end = row['dw_uw']
        ctype = row['type']
        lo = row.get('lo', 0)
        ro = row.get('ro', 0)
        
        # Color Logic
        if 'Decrease' in ctype and start < end:
            color = COLORS['fan'] # Red for "Punished"
            style = '-'
            width = 3
        elif 'Increase' in ctype and start > end:
            color = COLORS['system'] # Green for "Saved"
            style = '-'
            width = 3
        elif 'No change' in ctype:
            color = COLORS['highlight'] # Orange
            style = '-'
            width = 4
        elif 'Conditional' in ctype:
            color = COLORS['system']
            style = '--'
            width = 2
        else: # Stability
            color = COLORS['neutral']
            style = ':'
            width = 2
            
        ax.plot([left_x, right_x], [start, end], color=color, linewidth=width, linestyle=style, marker='o', markersize=10)
        
        # Labels
        # Left (Official)
        ax.text(left_x - 0.02, start + lo, f"{start}. {name}", ha='right', va='center', fontsize=12, fontweight='bold', color=COLORS['text'])
        
        # Right (New)
        ax.text(right_x + 0.02, end + ro, f"{end}. {name}", ha='left', va='center', fontsize=12, fontweight='bold', color=COLORS['text'])
        
        # Annotation on line
        mid_y = (start + end) / 2
        ax.text(0.5, mid_y, ctype, ha='center', va='center', fontsize=10, 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # Axes
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(9, 0) # Invert Y, show top 8
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Official Result', 'DW-UW Result'], fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.grid(False)
    
    # Remove borders
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add Legend/Key manually
    ax.text(0.5, 8.5, "Legend:\n• Red: Rank decrease\n• Green: Rank increase\n• Orange: No change",
            ha='center', va='center', fontsize=11, bbox=dict(facecolor='#ECF0F1', edgecolor='gray', boxstyle='round,pad=1'))
        
    plt.title('Impact Spectrum', pad=20)
    
    path = os.path.join(FIG_DIR, "Q4_fig4_impact_spectrum.png")
    plt.savefig(path)
    plt.close()

# -------------------------
# 5. Execution
# -------------------------
if __name__ == "__main__":
    plot_adaptive_mechanism()
    plot_tradeoff_quadrant()
    plot_suspense_violin()
    plot_impact_spectrum()
    print("Visual Upgrade Complete.")
