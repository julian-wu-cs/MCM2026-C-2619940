"""
MCM/ICM 2026 Problem C - Question 3
Random Forest modeling for audience and judges.

Workflow:
1) Build weekly modeling data from the official dataset and Q1 outputs.
2) Train two models:
   - Fans model (target: fan_pct_mean)
   - Judges model (target: judge_pct)
3) Export CV metrics and permutation importance tables.
4) Generate SHAP figures through visual3.py.
"""

import re
import argparse
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

from project_paths import get_data_path, get_out_dir, ensure_dir
from visual3 import generate_q3_shap_figures

warnings.filterwarnings("ignore")


DEFAULT_Q1_PATH = get_out_dir("Q1", None) / "tables" / "Q1_1.csv"

RANDOM_STATE = 42

RF_PARAMS = dict(
    n_estimators=800,
    max_depth=None,
    min_samples_leaf=3,
    min_samples_split=8,
    max_features="sqrt",
    n_jobs=1,
    random_state=RANDOM_STATE,
)


def wide_to_weekly_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    judge_cols = [c for c in df_wide.columns if re.match(r"week\d+_judge\d+_score", c)]
    if not judge_cols:
        raise ValueError("No weekX_judgeY_score columns found.")

    df_m = df_wide.melt(
        id_vars=[
            "celebrity_name",
            "ballroom_partner",
            "celebrity_industry",
            "celebrity_homecountry/region",
            "celebrity_age_during_season",
            "season",
            "results",
            "placement",
        ],
        value_vars=judge_cols,
        var_name="week_judge",
        value_name="judge_score",
    )

    m = df_m["week_judge"].str.extract(r"week(?P<week>\d+)_judge(?P<judge>\d+)_score")
    df_m["week"] = m["week"].astype(int)
    df_m["judge_id"] = m["judge"].astype(int)
    df_m["judge_score"] = pd.to_numeric(df_m["judge_score"], errors="coerce")

    agg = (
        df_m.groupby(["season", "celebrity_name", "week"], as_index=False)
        .agg(
            judge_sum=("judge_score", "sum"),
            judge_mean=("judge_score", "mean"),
            judge_count=("judge_score", lambda x: int(np.sum(~pd.isna(x)))),
        )
    )

    static_cols = [
        "season", "celebrity_name",
        "ballroom_partner",
        "celebrity_industry",
        "celebrity_homecountry/region",
        "celebrity_age_during_season",
        "results",
        "placement",
    ]
    df_static = df_wide[static_cols].drop_duplicates(["season", "celebrity_name"])
    df_week = agg.merge(df_static, on=["season", "celebrity_name"], how="left")
    return df_week


def add_judge_pct_and_active_count(df_week: pd.DataFrame) -> pd.DataFrame:
    df = df_week.copy()
    df["is_active"] = df["judge_sum"] > 0

    active_count = (
        df[df["is_active"]]
        .groupby(["season", "week"])["celebrity_name"]
        .nunique()
        .rename("active_count")
        .reset_index()
    )
    df = df.merge(active_count, on=["season", "week"], how="left")

    df_active = df[df["is_active"]].copy()
    totals = (
        df_active.groupby(["season", "week"])["judge_sum"]
        .sum()
        .rename("week_judge_total")
        .reset_index()
    )
    df_active = df_active.merge(totals, on=["season", "week"], how="left")
    df_active["judge_pct"] = df_active["judge_sum"] / df_active["week_judge_total"]

    df = df.merge(
        df_active[["season", "celebrity_name", "week", "judge_pct"]],
        on=["season", "celebrity_name", "week"],
        how="left",
    )
    return df


def add_prev_judge_pct(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.sort_values(["season", "celebrity_name", "week"]).copy()
    g = df2.groupby(["season", "celebrity_name"], group_keys=False)
    df2["prev_judge_pct"] = g["judge_pct"].shift(1)
    return df2


def add_fan_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.sort_values(["season", "celebrity_name", "week"]).copy()
    g = df2.groupby(["season", "celebrity_name"], group_keys=False)

    df2["prev_fan_pct"] = g["fan_pct_mean"].shift(1)

    cum = g["fan_pct_mean"].expanding().mean()
    cum = cum.shift(1).reset_index(level=[0, 1], drop=True)
    df2["cummean_fan_pct"] = cum

    return df2


def load_q3_table(data_path: str, q1_path: str) -> pd.DataFrame:
    df_wide = pd.read_csv(data_path)
    q1 = pd.read_csv(q1_path)

    required_cols = {"season", "week", "celebrity_name", "fan_pct_mean"}
    if not required_cols.issubset(set(q1.columns)):
        raise ValueError(f"Q1 table is missing required columns: {required_cols - set(q1.columns)}")

    df_week = wide_to_weekly_long(df_wide)
    df_week = add_judge_pct_and_active_count(df_week)
    df_week = add_prev_judge_pct(df_week)

    df = df_week.merge(
        q1[["season", "week", "celebrity_name", "fan_pct_mean"]],
        on=["season", "week", "celebrity_name"],
        how="inner",
    )

    df = df[df["judge_sum"] > 0].copy()

    df = add_fan_history_features(df)

    df["celebrity_age_during_season"] = pd.to_numeric(df["celebrity_age_during_season"], errors="coerce")
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")

    return df


def build_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop"
    )

    model = RandomForestRegressor(**RF_PARAMS)

    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("rf", model),
    ])
    return pipe


def group_cv_metrics(pipe: Pipeline,
                     X: pd.DataFrame,
                     y: pd.Series,
                     groups: pd.Series,
                     n_splits: int = 5) -> Dict[str, float]:
    gkf = GroupKFold(n_splits=n_splits)
    r2_list, mae_list = [], []

    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)

        r2_list.append(r2_score(yte, pred))
        mae_list.append(mean_absolute_error(yte, pred))

    return {
        "cv_r2_mean": float(np.mean(r2_list)),
        "cv_r2_std": float(np.std(r2_list)),
        "cv_mae_mean": float(np.mean(mae_list)),
        "cv_mae_std": float(np.std(mae_list)),
    }


def compute_permutation_importance_on_raw_cols(pipe: Pipeline,
                                               X: pd.DataFrame,
                                               y: pd.Series,
                                               n_repeats: int = 10) -> pd.DataFrame:
    pipe.fit(X, y)
    r = permutation_importance(
        pipe, X, y,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    imp = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False)

    return imp


def main(data_path, q1_path, out_dir):
    output_fig_dir = ensure_dir(out_dir / "figs")
    output_table_dir = ensure_dir(out_dir / "tables")

    df = load_q3_table(str(data_path), str(q1_path))

    numeric_features = [
        "celebrity_age_during_season",
        "season",
        "week",
        "active_count",
        "judge_pct",
        "prev_judge_pct",
        "prev_fan_pct",
        "cummean_fan_pct",
    ]
    categorical_features = [
        "celebrity_industry",
        "celebrity_homecountry/region",
        "ballroom_partner",
    ]

    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    groups = df["season"]

    y_fans = df["fan_pct_mean"].copy()
    X_fans = df[numeric_features + categorical_features].copy()
    fans_pipe = build_pipeline(numeric_features, categorical_features)
    fans_metrics = group_cv_metrics(fans_pipe, X_fans, y_fans, groups, n_splits=5)
    fans_metrics["model"] = "Fans (fan_pct_mean)"

    fans_pipe.fit(X_fans, y_fans)
    fans_imp = compute_permutation_importance_on_raw_cols(fans_pipe, X_fans, y_fans, n_repeats=10)
    fans_imp.to_csv(str(output_table_dir / "permutation_importance_fans.csv"), index=False)

    y_judges = df["judge_pct"].copy()
    numeric_features_j = [
        "celebrity_age_during_season",
        "season",
        "week",
        "active_count",
        "prev_judge_pct",
    ]
    numeric_features_j = [c for c in numeric_features_j if c in df.columns]

    X_judges = df[numeric_features_j + categorical_features].copy()
    judges_pipe = build_pipeline(numeric_features_j, categorical_features)
    judges_metrics = group_cv_metrics(judges_pipe, X_judges, y_judges, groups, n_splits=5)
    judges_metrics["model"] = "Judges (judge_pct)"

    judges_pipe.fit(X_judges, y_judges)
    judges_imp = compute_permutation_importance_on_raw_cols(judges_pipe, X_judges, y_judges, n_repeats=10)
    judges_imp.to_csv(str(output_table_dir / "permutation_importance_judges.csv"), index=False)

    generate_q3_shap_figures(
        fans_pipe=fans_pipe,
        x_fans=X_fans,
        fans_numeric_features=numeric_features,
        common_categorical_features=categorical_features,
        judges_pipe=judges_pipe,
        x_judges=X_judges,
        judges_numeric_features=numeric_features_j,
        out_fig_dir=output_fig_dir,
    )

    metrics = pd.DataFrame([fans_metrics, judges_metrics])
    metrics.to_csv(str(output_table_dir / "metrics_cv.csv"), index=False)

    print("=== Q3 RF + SHAP Finished ===")
    print(f"[Saved] {str(output_table_dir / 'metrics_cv.csv')}")
    print(f"[Saved] {str(output_table_dir / 'permutation_importance_fans.csv')}")
    print(f"[Saved] {str(output_table_dir / 'permutation_importance_judges.csv')}")
    print(f"[Saved] SHAP figures in: {str(output_fig_dir)}")
    print("\nCV Metrics Preview:")
    print(metrics)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Q3")
    ap.add_argument("--data", type=str, default=None, help="Path to data csv")
    ap.add_argument("--out", type=str, default=None, help="Output directory")
    args = ap.parse_args()

    data_path = get_data_path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found: {data_path}")
    q1_path = DEFAULT_Q1_PATH
    if not q1_path.exists():
        raise FileNotFoundError(f"Q1 table not found: {q1_path}")
    out_dir = ensure_dir(get_out_dir("Q3", args.out))

    main(data_path=data_path, q1_path=q1_path, out_dir=out_dir)
