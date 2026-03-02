from pathlib import Path
import re
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from project_paths import ensure_dir


RANDOM_STATE = 42


def get_ohe_feature_names(
    preprocess: ColumnTransformer,
    numeric_features: List[str],
    categorical_features: List[str],
) -> List[str]:
    num_names = numeric_features
    ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(categorical_features).tolist()
    return num_names + cat_names


def shap_analysis_and_plots(
    pipe: Pipeline,
    X: pd.DataFrame,
    y_name: str,
    out_prefix: str,
    numeric_features: List[str],
    categorical_features: List[str],
    out_fig_dir: Path,
    max_samples: int = 2000,
    top_dependence: int = 3,
) -> None:
    preprocess = pipe.named_steps["preprocess"]
    rf = pipe.named_steps["rf"]

    X_enc = preprocess.transform(X)
    n = X_enc.shape[0]
    if n > max_samples:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(n, size=max_samples, replace=False)
        X_enc_s = X_enc[idx]
    else:
        X_enc_s = X_enc

    if hasattr(X_enc_s, "toarray"):
        X_enc_s = X_enc_s.toarray()
    X_enc_s = np.asarray(X_enc_s, dtype=np.float64)

    feature_names = get_ohe_feature_names(preprocess, numeric_features, categorical_features)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_enc_s)

    plt.figure()
    shap.summary_plot(shap_values, X_enc_s, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary (Beeswarm) - {y_name}")
    plt.tight_layout()
    plt.savefig(str(out_fig_dir / f"{out_prefix}_shap_summary_beeswarm.png"), dpi=300)
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_enc_s,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.title(f"SHAP Feature Importance (Bar) - {y_name}")
    plt.tight_layout()
    plt.savefig(str(out_fig_dir / f"{out_prefix}_shap_summary_bar.png"), dpi=300)
    plt.close()

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_dependence]
    top_features = [feature_names[i] for i in top_idx]

    for feature in top_features:
        plt.figure()
        shap.dependence_plot(
            feature,
            shap_values,
            X_enc_s,
            feature_names=feature_names,
            show=False,
        )
        plt.title(f"SHAP Dependence - {y_name} - {feature}")
        plt.tight_layout()
        safe_feature = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", feature)[:80]
        plt.savefig(str(out_fig_dir / f"{out_prefix}_shap_dependence_{safe_feature}.png"), dpi=300)
        plt.close()


def generate_q3_shap_figures(
    fans_pipe: Pipeline,
    x_fans: pd.DataFrame,
    fans_numeric_features: List[str],
    common_categorical_features: List[str],
    judges_pipe: Pipeline,
    x_judges: pd.DataFrame,
    judges_numeric_features: List[str],
    out_fig_dir: Path,
) -> Path:
    out_fig_dir = ensure_dir(out_fig_dir)

    shap_analysis_and_plots(
        pipe=fans_pipe,
        X=x_fans,
        y_name="Fans (fan_pct_mean)",
        out_prefix="fans",
        numeric_features=fans_numeric_features,
        categorical_features=common_categorical_features,
        out_fig_dir=out_fig_dir,
        max_samples=2000,
        top_dependence=3,
    )

    shap_analysis_and_plots(
        pipe=judges_pipe,
        X=x_judges,
        y_name="Judges (judge_pct)",
        out_prefix="judges",
        numeric_features=judges_numeric_features,
        categorical_features=common_categorical_features,
        out_fig_dir=out_fig_dir,
        max_samples=2000,
        top_dependence=3,
    )

    return out_fig_dir

