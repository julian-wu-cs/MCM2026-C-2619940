import os
import re
import math
import argparse
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from project_paths import get_data_path, get_out_dir, ensure_dir




RANK_CLASSIC_SEASONS = set([1, 2])          # 1rank_classic
PCT_SEASONS = set(range(3, 28))             # 2pct
BOTTOM2_SAVE_SEASONS = set(range(28, 35))   # 3bottom2_save28


TEST_MODE = False
TEST_RANK_CLASSIC = [1, 2]
TEST_PCT = [3, 4]
TEST_BOTTOM2_SAVE = [28, 29]
TEST_SEASONS_SET = set(TEST_RANK_CLASSIC + TEST_PCT + TEST_BOTTOM2_SAVE)


SIGMA_U0 = 0.5
SIGMA_LEVEL = 0.25
SIGMA_RW = 0.02
TEMP_SOFTMAX = 0.25

USE_JUDGE_ANCHOR = True
SIGMA_ANCHOR = 0.10
ANCHOR_ALPHA = 2.5

TAU_PCT = 0.002
TAU_RANK = 0.2

BETA_BOTTOM2 = 1.5
ETA_SAVE = 0.20
TAU_SAVE = 2.0

TAU_FINAL = 0.05
LAMBDA_FINAL = 5.0


# MCMC 
if TEST_MODE:
    N_SAMPLES = 1500
    BURN = 800
    THIN = 2
    STEP_WEEK_INIT = 0.10
else:
    N_SAMPLES = 10000
    BURN = 5000
    THIN = 5
    STEP_WEEK_INIT = 0.12

SEED = 2026

ADAPT_STEP = True
ADAPT_TARGET = 0.25
ADAPT_INTERVAL = 25
ADAPT_RATE = 0.6
STEP_MIN = 0.02
STEP_MAX = 0.30


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    ez = np.exp(z)
    s = ez.sum()
    return ez / s if s > 0 else np.ones_like(z) / len(z)


def softmax_temp(u: np.ndarray, temp: float) -> np.ndarray:
    t = max(float(temp), 1e-6)
    return softmax(u / t)


def logsumexp(arr: np.ndarray) -> float:
    m = float(np.max(arr))
    return float(m + np.log(np.sum(np.exp(arr - m))))


def ranks_from_scores(scores: np.ndarray, higher_better: bool = True) -> np.ndarray:
    if len(scores) == 0:
        return np.array([], dtype=float)
    order = np.argsort(-scores) if higher_better else np.argsort(scores)
    ranks = np.empty_like(order, dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks.astype(float)


def parse_withdrew(res) -> bool:
    if pd.isna(res):
        return False
    s = str(res).lower()
    keywords = ["withdrew", "withdrawn", "withdraw", "quit", "medical", "injury"]
    return any(k in s for k in keywords)


def parse_place_rank_from_text(s: str) -> Optional[int]:
    if not s:
        return None
    t = str(s).lower()
    if re.search(r"\bwinner\b", t):
        return 1
    m = re.search(r"\b(\d+)\s*(st|nd|rd|th)\s*place\b", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def parse_placement_rank(x) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()

    v = pd.to_numeric(s, errors="coerce")
    if pd.notna(v):
        iv = int(v)
        if iv > 0:
            return iv

    m = re.match(r"^\s*(\d+)\s*(st|nd|rd|th)\s*$", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None

    if "winner" in s:
        return 1
    return None


#  exit_week / exit_type
def build_week_table(df: pd.DataFrame) -> pd.DataFrame:
    score_cols = [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]
    if not score_cols:
        raise ValueError("No week*_judge*_score columns found.")

    id_vars = [
        "season", "celebrity_name", "results", "placement",
        "celebrity_industry", "celebrity_homestate", "celebrity_homecountry/region",
        "celebrity_age_during_season", "ballroom_partner"
    ]
    id_vars = [c for c in id_vars if c in df.columns]

    long = df.melt(
        id_vars=id_vars,
        value_vars=score_cols,
        var_name="wk_j",
        value_name="score"
    )
    long["week"] = long["wk_j"].str.extract(r"week(\d+)_")[0].astype(int)
    long.drop(columns=["wk_j"], inplace=True)
    long["score"] = pd.to_numeric(long["score"], errors="coerce").fillna(0.0)

    agg = (
        long.groupby(["season", "week", "celebrity_name"], as_index=False)
            .agg(judge_total=("score", "sum"))
    )

    wk = agg[agg["judge_total"] > 0].copy()

    meta_cols = [c for c in id_vars if c != "week"]
    meta = df[meta_cols].copy()

    if "results" in meta.columns:
        meta["withdrew_flag"] = meta["results"].apply(parse_withdrew)
        meta["rank_from_results"] = meta["results"].astype(str).apply(parse_place_rank_from_text)
    else:
        meta["withdrew_flag"] = False
        meta["rank_from_results"] = None

    if "placement" in meta.columns:
        meta["rank_from_placement"] = meta["placement"].apply(parse_placement_rank)
    else:
        meta["rank_from_placement"] = None

    meta["final_rank"] = meta["rank_from_placement"]
    miss = meta["final_rank"].isna()
    meta.loc[miss, "final_rank"] = meta.loc[miss, "rank_from_results"]

    season_final_week = wk.groupby("season")["week"].max().rename("season_final_week").reset_index()
    meta = meta.merge(season_final_week, on="season", how="left")

    active = (
        wk.groupby(["season", "week"])["celebrity_name"]
          .apply(lambda x: set(x.tolist()))
          .reset_index()
          .rename(columns={"celebrity_name": "active_set"})
    )

    wd_lookup = {(int(r.season), str(r.celebrity_name)): bool(r.withdrew_flag) for r in meta.itertuples()}

    exit_week_map: Dict[Tuple[int, str], int] = {}
    exit_type_map: Dict[Tuple[int, str], str] = {}

    for season, g in active.groupby("season"):
        season = int(season)
        weeks = sorted(g["week"].tolist())
        sets = {int(w): s for w, s in zip(g["week"].tolist(), g["active_set"].tolist())}
        if not weeks:
            continue

        for i in range(len(weeks) - 1):
            w = int(weeks[i])
            w_next = int(weeks[i + 1])
            dropped = sets[w] - sets[w_next]
            for name in dropped:
                key = (season, str(name))
                if key not in exit_week_map:
                    exit_week_map[key] = w
                    exit_type_map[key] = "withdrew" if wd_lookup.get(key, False) else "eliminated"

        final_w = int(weeks[-1])
        finalists = sets[final_w]

        meta_s = meta[meta["season"] == season][["celebrity_name", "final_rank"]].copy()
        rank_dict = {str(a): (None if pd.isna(b) else int(b))
                     for a, b in zip(meta_s["celebrity_name"], meta_s["final_rank"])}

        winner = None
        for nm in finalists:
            if rank_dict.get(str(nm), None) == 1:
                winner = str(nm)
                break

        for name in finalists:
            key = (season, str(name))
            if winner is not None and str(name) == winner:
                continue
            if key not in exit_week_map:
                exit_week_map[key] = final_w
                exit_type_map[key] = "eliminated"

    meta["exit_week"] = meta.apply(
        lambda r: exit_week_map.get((int(r["season"]), str(r["celebrity_name"])), np.nan),
        axis=1
    )
    meta["exit_type"] = meta.apply(
        lambda r: exit_type_map.get((int(r["season"]), str(r["celebrity_name"])), None),
        axis=1
    )

    keep_meta_cols = [c for c in meta.columns if c not in ["rank_from_results", "rank_from_placement"]]
    wk = wk.merge(meta[keep_meta_cols], on=["season", "celebrity_name"], how="left")

    week_sum = wk.groupby(["season", "week"])["judge_total"].transform("sum")
    wk["judge_pct"] = np.where(week_sum > 0, wk["judge_total"] / week_sum, 0.0)

    return wk.reset_index(drop=True)


def season_to_rule(season: int) -> str:
    if season in RANK_CLASSIC_SEASONS:
        return "rank_classic"
    if season in PCT_SEASONS:
        return "pct"
    if season in BOTTOM2_SAVE_SEASONS:
        return "bottom2_save"
    return "unknown"


# B-2 likelihood p
def ll_pct_elim_from_p(p: np.ndarray,
                       judge_pct: np.ndarray,
                       elim_mask: np.ndarray,
                       keep_mask: np.ndarray,
                       tau: float) -> float:
    c = judge_pct + p
    elim_idx = np.where(elim_mask)[0]
    keep_idx = np.where(keep_mask)[0]
    if len(elim_idx) == 0 or len(keep_idx) == 0:
        return 0.0
    t = max(float(tau), 1e-12)
    ll = 0.0
    for e in elim_idx:
        diffs = (c[keep_idx] - c[e]) / t
        probs = sigmoid(diffs)
        probs = np.clip(probs, 1e-12, 1.0)
        ll += float(np.sum(np.log(probs)))
    return float(ll)


def ll_rank_elim_from_p(p: np.ndarray,
                        judge_total: np.ndarray,
                        elim_mask: np.ndarray,
                        keep_mask: np.ndarray,
                        tau_rank: float) -> float:
    fan_rank = ranks_from_scores(p, higher_better=True)
    judge_rank = ranks_from_scores(judge_total, higher_better=True)
    combined = judge_rank + fan_rank + 0.01 * fan_rank

    elim_idx = np.where(elim_mask)[0]
    keep_idx = np.where(keep_mask)[0]
    if len(elim_idx) == 0 or len(keep_idx) == 0:
        return 0.0

    t = max(float(tau_rank), 1e-12)
    ll = 0.0
    for e in elim_idx:
        diffs = (combined[e] - combined[keep_idx]) / t
        probs = sigmoid(diffs)
        probs = np.clip(probs, 1e-12, 1.0)
        ll += float(np.sum(np.log(probs)))
    return float(ll)


def ll_bottom2_save_from_p(p: np.ndarray,
                           judge_total: np.ndarray,
                           elim_mask: np.ndarray,
                           keep_mask: np.ndarray,
                           beta_bottom2: float,
                           eta_save: float,
                           tau_save: float) -> float:
    fan_rank = ranks_from_scores(p, higher_better=True)
    judge_rank = ranks_from_scores(judge_total, higher_better=True)
    combined = judge_rank + fan_rank  # 

    eligible = np.where(elim_mask | keep_mask)[0]
    if len(eligible) < 2:
        return 0.0

    bad = combined[eligible].astype(float)
    bad = bad - bad.mean()
    w = np.exp(beta_bottom2 * bad)
    sumw = float(w.sum())

    elim_idx = np.where(elim_mask)[0]
    if len(elim_idx) == 0:
        return 0.0

    ll = 0.0
    for e in elim_idx:
        if e not in eligible:
            continue
        e_pos = int(np.where(eligible == e)[0][0])
        we = float(w[e_pos])

        log_terms = []
        for j_pos, j in enumerate(eligible):
            if j == e:
                continue
            wj = float(w[j_pos])

            p_ej = (we / sumw) * (wj / (sumw - we))
            p_je = (wj / sumw) * (we / (sumw - wj))
            p_pair = max(p_ej + p_je, 1e-12)

            t = max(float(tau_save), 1e-12)
            p_by_judge = float(sigmoid(np.array([(judge_total[j] - judge_total[e]) / t]))[0])
            p_elim_given_pair = (1.0 - eta_save) * 0.5 + eta_save * p_by_judge
            p_elim_given_pair = float(np.clip(p_elim_given_pair, 1e-12, 1.0))

            log_terms.append(math.log(p_pair) + math.log(p_elim_given_pair))

        if len(log_terms) > 0:
            ll += logsumexp(np.array(log_terms, dtype=float))

    return float(ll)


def ll_final_rank_plackett_luce_from_p(p: np.ndarray,
                                       season_rule: str,
                                       judge_total: np.ndarray,
                                       judge_pct: np.ndarray,
                                       ranks: np.ndarray,
                                       tau_final: float) -> float:
    idx = np.where(np.isfinite(ranks))[0]
    if len(idx) < 2:
        return 0.0

    if season_rule in ["pct", "bottom2_save"]:
        score = judge_pct + p
    else:
        fan_rank = ranks_from_scores(p, higher_better=True)
        judge_rank = ranks_from_scores(judge_total, higher_better=True)
        combined = judge_rank + fan_rank + 0.01 * fan_rank
        score = -combined

    order = idx[np.argsort(ranks[idx])]

    t = max(float(tau_final), 1e-12)
    ll = 0.0
    remaining = order.copy()
    for _ in range(len(order) - 1):
        chosen = int(remaining[0])
        rem_scores = score[remaining] / t
        ll += float(score[chosen] / t - logsumexp(rem_scores))
        remaining = remaining[1:]
    return float(ll)


def final_rank_margins(samples_mat: np.ndarray,
                       season_rule: str,
                       judge_total: np.ndarray,
                       judge_pct: np.ndarray,
                       ranks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    S, n = samples_mat.shape
    idx = np.where(np.isfinite(ranks))[0]
    if len(idx) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    order = idx[np.argsort(ranks[idx])]
    pairs_all = [(int(order[a]), int(order[b]))
                 for a in range(len(order)) for b in range(a + 1, len(order))]
    pairs_adj = [(int(order[a]), int(order[a + 1])) for a in range(len(order) - 1)]

    if season_rule not in ["pct", "bottom2_save"]:
        judge_rank = ranks_from_scores(judge_total, higher_better=True)

    all_m = []
    adj_m = []
    for s in range(S):
        p = samples_mat[s]
        if season_rule in ["pct", "bottom2_save"]:
            score = judge_pct + p
        else:
            fan_rank = ranks_from_scores(p, higher_better=True)
            combined = judge_rank + fan_rank + 0.01 * fan_rank
            score = -combined

        ms_all = [float(score[i] - score[j]) for (i, j) in pairs_all]
        ms_adj = [float(score[i] - score[j]) for (i, j) in pairs_adj]
        all_m.append(float(np.min(ms_all)) if ms_all else np.nan)
        adj_m.append(float(np.min(ms_adj)) if ms_adj else np.nan)

    return np.asarray(all_m, dtype=float), np.asarray(adj_m, dtype=float)


def week_diagnostics_from_samples(samples_mat: np.ndarray,
                                 season_rule: str,
                                 mode: str,
                                 judge_total: np.ndarray,
                                 judge_pct: np.ndarray,
                                 elim_mask: np.ndarray,
                                 keep_mask: np.ndarray,
                                 final_ranks_used: np.ndarray) -> Dict[str, float]:
    out = {
        "consistency_prob": np.nan,
        "margin_mean": np.nan,
        "margin_sd": np.nan,
        "margin_ci90_width": np.nan,

        "consistency_allpairs": np.nan,
        "margin_allpairs_mean": np.nan,
        "margin_allpairs_sd": np.nan,
        "margin_allpairs_ci90_width": np.nan,

        "consistency_adjacent": np.nan,
        "margin_adjacent_mean": np.nan,
        "margin_adjacent_sd": np.nan,
        "margin_adjacent_ci90_width": np.nan,
    }

    if mode == "none":
        return out

    if mode == "pct_elim":
        elim_idx = np.where(elim_mask)[0]
        keep_idx = np.where(keep_mask)[0]
        if len(elim_idx) == 0 or len(keep_idx) == 0:
            return out
        c = judge_pct[None, :] + samples_mat
        elim_c = c[:, elim_idx]
        keep_c = c[:, keep_idx]
        margins = np.min(keep_c, axis=1) - np.max(elim_c, axis=1)
        out["consistency_prob"] = float(np.mean(margins >= 0.0))
        out["margin_mean"] = float(np.mean(margins))
        q05, q95 = np.quantile(margins, [0.05, 0.95])
        out["margin_ci90_width"] = float(q95 - q05)
        return out

    if mode == "rank_elim":
        elim_idx = np.where(elim_mask)[0]
        keep_idx = np.where(keep_mask)[0]
        if len(elim_idx) == 0 or len(keep_idx) == 0:
            return out
        judge_rank = ranks_from_scores(judge_total, higher_better=True)
        mlist = []
        for s in range(samples_mat.shape[0]):
            p = samples_mat[s]
            fan_rank = ranks_from_scores(p, higher_better=True)
            combined = judge_rank + fan_rank + 0.01 * fan_rank
            diffs = []
            for e in elim_idx:
                diffs.append(np.min(combined[e] - combined[keep_idx]))
            mlist.append(float(np.min(diffs)))
        margins = np.asarray(mlist, dtype=float)
        out["consistency_prob"] = float(np.mean(margins >= 0.0))
        out["margin_mean"] = float(np.mean(margins))
        q05, q95 = np.quantile(margins, [0.05, 0.95])
        out["margin_ci90_width"] = float(q95 - q05)
        return out

    if mode == "bottom2_save":
        elim_idx = np.where(elim_mask)[0]
        eligible = np.where(elim_mask | keep_mask)[0]
        if len(elim_idx) == 0 or len(eligible) < 2:
            return out
        e = int(elim_idx[0])
        judge_rank = ranks_from_scores(judge_total, higher_better=True)
        mlist = []
        for s in range(samples_mat.shape[0]):
            p = samples_mat[s]
            fan_rank = ranks_from_scores(p, higher_better=True)
            combined = judge_rank + fan_rank
            vals = combined[eligible]
            second_worst = np.sort(vals)[-2]
            mlist.append(float(combined[e] - second_worst))
        margins = np.asarray(mlist, dtype=float)
        out["consistency_prob"] = float(np.mean(margins >= 0.0))
        out["margin_mean"] = float(np.mean(margins))
        q05, q95 = np.quantile(margins, [0.05, 0.95])
        out["margin_ci90_width"] = float(q95 - q05)
        return out

    if mode == "final_rank":
        m_all, m_adj = final_rank_margins(
            samples_mat=samples_mat,
            season_rule=season_rule,
            judge_total=judge_total,
            judge_pct=judge_pct,
            ranks=final_ranks_used
        )
        if len(m_adj) == 0:
            return out
        out["consistency_adjacent"] = float(np.mean(m_adj >= 0.0))
        out["margin_adjacent_mean"] = float(np.mean(m_adj))
        q05, q95 = np.quantile(m_adj, [0.05, 0.95])
        out["margin_adjacent_ci90_width"] = float(q95 - q05)

        out["consistency_prob"] = out["consistency_adjacent"]
        out["margin_mean"] = out["margin_adjacent_mean"]
        out["margin_ci90_width"] = out["margin_adjacent_ci90_width"]
        return out

    return out


#    fan_ci90_width fan_p05 / fan_p95
def summarize_uncertainty_metrics(samples_p_1d: np.ndarray,
                                  baseline_uniform: float) -> Dict[str, float]:
    mu = float(np.mean(samples_p_1d))
    sd = float(np.std(samples_p_1d, ddof=0))
    q05, q95 = np.quantile(samples_p_1d, [0.05, 0.95])

    ci90 = float(q95 - q05)  #  rci90_pct

    denom = max(mu, baseline_uniform, 0.005)
    rsd_pct = 100.0 * sd / denom
    rci90_pct = 100.0 * ci90 / denom

    return {
        "fan_pct_mean": float(mu),
        "fan_p05": float(q05),
        "fan_p95": float(q95),
        "rsd_pct": float(rsd_pct),
        "rci90_pct": float(rci90_pct),
    }


# B-2blocked MH by week
def dynamic_mh_season(weeks_sorted: List[int],
                      week_obs: List[Dict],
                      n_samples: int,
                      burn: int,
                      thin: int,
                      sigma_u0: float,
                      sigma_level: float,
                      sigma_rw: float,
                      step_week_init: float,
                      seed: int) -> Tuple[List[np.ndarray], Dict[str, float]]:
    rng = np.random.default_rng(seed)
    W = len(weeks_sorted)

    week_name_to_pos = []
    for obs in week_obs:
        mp = {str(nm): i for i, nm in enumerate(obs["names"])}
        week_name_to_pos.append(mp)

    U = []
    for obs in week_obs:
        n = len(obs["names"])
        u = rng.normal(0.0, 0.01, size=n)
        u = u - u.mean()
        U.append(u)

    su2 = max(float(sigma_u0), 1e-12) ** 2
    sl2 = max(float(sigma_level), 1e-12) ** 2
    srw2 = max(float(sigma_rw), 1e-12) ** 2
    sa2 = max(float(SIGMA_ANCHOR), 1e-12) ** 2

    def make_anchor_u(judge_pct: np.ndarray) -> np.ndarray:
        v = np.log(np.clip(judge_pct, 1e-8, 1.0))
        v = v - np.mean(v)
        v = ANCHOR_ALPHA * v
        return v

    def log_prior_week(t: int, u_t: np.ndarray, u_prev: Optional[np.ndarray]) -> float:
        lp = 0.0
        lp += float(-0.5 * np.sum(u_t ** 2) / sl2)

        if t == 0:
            lp += float(-0.5 * np.sum(u_t ** 2) / su2)

        if t > 0 and (u_prev is not None):
            mp_prev = week_name_to_pos[t - 1]
            mp_cur = week_name_to_pos[t]
            common = set(mp_prev.keys()) & set(mp_cur.keys())
            if common:
                diffs = np.asarray([u_t[mp_cur[nm]] - u_prev[mp_prev[nm]] for nm in common], dtype=float)
                lp += float(-0.5 * np.sum(diffs ** 2) / srw2)

        if USE_JUDGE_ANCHOR:
            is_final = bool(week_obs[t]["is_final_week"])
            has_constr = bool(week_obs[t]["has_constraint"])
            if (not is_final) and (not has_constr):
                anchor_u = make_anchor_u(week_obs[t]["judge_pct"])
                lp += float(-0.5 * np.sum((u_t - anchor_u) ** 2) / sa2)

        return float(lp)

    def log_like_week(t: int, u_t: np.ndarray) -> float:
        obs = week_obs[t]
        p = softmax_temp(u_t, TEMP_SOFTMAX)

        mode = obs["mode"]
        rule = obs["season_rule"]

        if mode == "pct_elim":
            return ll_pct_elim_from_p(p, obs["judge_pct"], obs["elim_mask"], obs["keep_mask"], TAU_PCT)
        if mode == "rank_elim":
            return ll_rank_elim_from_p(p, obs["judge_total"], obs["elim_mask"], obs["keep_mask"], TAU_RANK)
        if mode == "bottom2_save":
            return ll_bottom2_save_from_p(
                p, obs["judge_total"], obs["elim_mask"], obs["keep_mask"],
                BETA_BOTTOM2, ETA_SAVE, TAU_SAVE
            )
        if mode == "final_rank":
            return LAMBDA_FINAL * ll_final_rank_plackett_luce_from_p(
                p, rule, obs["judge_total"], obs["judge_pct"], obs["final_ranks_used"], TAU_FINAL
            )
        return 0.0

    cur_ll = [log_like_week(t, U[t]) for t in range(W)]
    step = float(np.clip(float(step_week_init), STEP_MIN, STEP_MAX))

    accepts = 0
    total_props = 0
    samples_by_week: List[List[np.ndarray]] = [[] for _ in range(W)]
    total_iters = burn + n_samples * thin

    window_acc = 0
    window_props = 0

    print(f"  > MCMC Sampling (Season with {W} weeks)... Total iters: {total_iters}")

    for it in range(total_iters):
        if (it + 1) % 1000 == 0:
            print(f"    Iter {it + 1}/{total_iters} | Accept Rate: {accepts/max(total_props, 1):.3f} | Step: {step:.4f}")

        for t in range(W):
            total_props += 1
            window_props += 1

            u_old = U[t]
            n = len(u_old)

            u_prop = u_old + rng.normal(0.0, step, size=n)
            u_prop = u_prop - u_prop.mean()

            ll_old = cur_ll[t]
            ll_prop = log_like_week(t, u_prop)

            lp_old = 0.0
            lp_prop = 0.0

            u_prev = U[t - 1] if t - 1 >= 0 else None
            lp_old += log_prior_week(t, U[t], u_prev)
            lp_prop += log_prior_week(t, u_prop, u_prev)

            if t + 1 < W:
                u_next = U[t + 1]
                lp_old += log_prior_week(t + 1, u_next, U[t])
                lp_prop += log_prior_week(t + 1, u_next, u_prop)

            log_alpha = (ll_prop - ll_old) + (lp_prop - lp_old)

            if np.log(rng.random()) < log_alpha:
                U[t] = u_prop
                cur_ll[t] = ll_prop
                accepts += 1
                window_acc += 1

        if ADAPT_STEP and (it < burn) and ((it + 1) % ADAPT_INTERVAL == 0):
            if window_props > 0:
                acc_rate_window = window_acc / window_props
                step = step * math.exp(ADAPT_RATE * (acc_rate_window - ADAPT_TARGET))
                step = float(np.clip(step, STEP_MIN, STEP_MAX))
            window_acc = 0
            window_props = 0

        if it >= burn and ((it - burn) % thin == 0):
            for t in range(W):
                samples_by_week[t].append(softmax_temp(U[t], TEMP_SOFTMAX))

    samples_out = [np.asarray(lst, dtype=float) for lst in samples_by_week]
    info = {
        "accept_rate": float(accepts / max(total_props, 1)),
        "final_step_week": float(step),
    }
    return samples_out, info


#  season 
def solve_q1_three_rules_dynamic(df: pd.DataFrame,
                                n_samples: int = N_SAMPLES,
                                burn: int = BURN,
                                thin: int = THIN,
                                sigma_u0: float = SIGMA_U0,
                                sigma_level: float = SIGMA_LEVEL,
                                sigma_rw: float = SIGMA_RW,
                                step_week_init: float = STEP_WEEK_INIT,
                                seed: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:

    wk = build_week_table(df)

    if TEST_MODE:
        wk = wk[wk["season"].astype(int).isin(TEST_SEASONS_SET)].copy()

    per_rows = []
    diag_rows = []

    for season, wk_s in wk.groupby("season"):
        season = int(season)
        print(f"\nProcessing Season {season}...")
        rule = season_to_rule(season)
        if rule == "unknown":
            print(f"  Skipping unknown rule season {season}")
            continue

        if TEST_MODE:
            if rule == "rank_classic" and season not in TEST_RANK_CLASSIC:
                continue
            if rule == "pct" and season not in TEST_PCT:
                continue
            if rule == "bottom2_save" and season not in TEST_BOTTOM2_SAVE:
                continue

        weeks_sorted = sorted(wk_s["week"].unique().astype(int).tolist())

        week_obs = []
        for week in weeks_sorted:
            g = wk_s[wk_s["week"].astype(int) == int(week)].copy()
            if g.empty:
                continue

            names = g["celebrity_name"].astype(str).values
            judge_total = g["judge_total"].values.astype(float)  # 
            judge_pct = g["judge_pct"].values.astype(float)

            exit_week = g["exit_week"].fillna(-1).astype(int).values
            exit_type = g["exit_type"].fillna("").astype(str).values

            elim_mask = (exit_week == week) & (exit_type == "eliminated")
            wd_mask = (exit_week == week) & (exit_type == "withdrew")
            keep_mask = (~elim_mask) & (~wd_mask)

            n_elim = int(np.sum(elim_mask))
            n_keep_used = int(np.sum(keep_mask))

            season_final_week = g["season_final_week"].iloc[0]
            is_final_week = (pd.notna(season_final_week) and int(season_final_week) == week)

            final_ranks = g["final_rank"].values.astype(float)
            final_ranks_used = final_ranks.copy()
            final_ranks_used[wd_mask] = np.nan
            n_ranked_used = int(np.sum(np.isfinite(final_ranks_used)))

            mode = "none"
            has_constraint = False
            if is_final_week and n_ranked_used >= 2:
                mode = "final_rank"
                has_constraint = True
            else:
                if n_elim > 0 and n_keep_used > 0:
                    if rule == "pct":
                        mode = "pct_elim"
                    elif rule == "rank_classic":
                        mode = "rank_elim"
                    else:
                        mode = "bottom2_save"
                    has_constraint = True

            week_obs.append({
                "season": season,
                "week": int(week),
                "season_rule": rule,
                "mode": mode,
                "has_constraint": bool(has_constraint),

                "names": names,
                "judge_total": judge_total,
                "judge_pct": judge_pct,

                "exit_week": exit_week,
                "exit_type": exit_type,
                "elim_mask": elim_mask,
                "wd_mask": wd_mask,
                "keep_mask": keep_mask,

                "final_ranks_used": final_ranks_used,

                "is_final_week": bool(is_final_week),
            })

        if not week_obs:
            continue

        samples_by_week, _info = dynamic_mh_season(
            weeks_sorted=[obs["week"] for obs in week_obs],
            week_obs=week_obs,
            n_samples=n_samples,
            burn=burn,
            thin=thin,
            sigma_u0=sigma_u0,
            sigma_level=sigma_level,
            sigma_rw=sigma_rw,
            step_week_init=step_week_init,
            seed=(seed + season * 1000)
        )

        for t, obs in enumerate(week_obs):
            samples_mat = samples_by_week[t]  # (S, n_active)
            _, n_active = samples_mat.shape
            baseline_uniform = 1.0 / max(n_active, 1)

            # CI90 B
            q05 = np.quantile(samples_mat, 0.05, axis=0)
            q95 = np.quantile(samples_mat, 0.95, axis=0)
            ci90w_week = q95 - q05
            week_fan_ci90_mean = float(np.mean(ci90w_week))

            diag = week_diagnostics_from_samples(
                samples_mat=samples_mat,
                season_rule=obs["season_rule"],
                mode=obs["mode"],
                judge_total=obs["judge_total"],
                judge_pct=obs["judge_pct"],
                elim_mask=obs["elim_mask"],
                keep_mask=obs["keep_mask"],
                final_ranks_used=obs["final_ranks_used"]
            )

            # === A fan_ci90_width fan_p05 / fan_p95===
            rci90_list = []

            for i, nm in enumerate(obs["names"]):
                samp_i = samples_mat[:, i]
                met = summarize_uncertainty_metrics(
                    samples_p_1d=samp_i,
                    baseline_uniform=baseline_uniform
                )
                rci90_list.append(float(met["rci90_pct"]))
                exit_w_i = int(obs["exit_week"][i])
                exit_t_i = str(obs["exit_type"][i]) if obs["exit_type"][i] else ""
                elim_this_week = int((exit_w_i == obs["week"]) and (exit_t_i == "eliminated"))

                per_rows.append({
                    "season": obs["season"],
                    "week": obs["week"],
                    "season_rule": obs["season_rule"],
                    "celebrity_name": str(nm),

                    "exit_type": (obs["exit_type"][i] if obs["exit_type"][i] else None),
                    "exit_week": int(obs["exit_week"][i]) if int(obs["exit_week"][i]) >= 0 else None,
                    "elim_this_week": int(elim_this_week),

                    "judge_pct": float(obs["judge_pct"][i]),
                    "fan_pct_mean": float(met["fan_pct_mean"]),
                    "fan_p05": float(met["fan_p05"]),
                    "fan_p95": float(met["fan_p95"]),
                    "rsd_pct": float(met["rsd_pct"]),
                    "rci90_pct": float(met["rci90_pct"]),
                })

            # === B===
            n_elim = int(np.sum(obs["elim_mask"]))
            n_withdrew = int(np.sum(obs["wd_mask"]))
            has_elim = int(n_elim > 0)
            has_withdrew = int(n_withdrew > 0)

            U_week_rel_rci90_mean = float(np.mean(rci90_list)) if len(rci90_list) > 0 else np.nan

            judge_pct_vec = np.asarray(obs["judge_pct"], dtype=float)
            judge_gap = float(np.max(judge_pct_vec) - np.min(judge_pct_vec)) if len(judge_pct_vec) > 0 else np.nan
            judge_concentration = float(np.sum(judge_pct_vec ** 2)) if len(judge_pct_vec) > 0 else np.nan

            diag_rows.append({
                "season": obs["season"],
                "week": obs["week"],
                "season_rule": obs["season_rule"],

                "consistency_prob": float(diag["consistency_prob"]) if pd.notna(diag["consistency_prob"]) else np.nan,
                "margin_mean": float(diag["margin_mean"]) if pd.notna(diag["margin_mean"]) else np.nan,
                "margin_ci90_width": float(diag["margin_ci90_width"]) if pd.notna(diag["margin_ci90_width"]) else np.nan,

                "week_fan_ci90_mean": float(week_fan_ci90_mean),

                "n_active": int(n_active),
                "n_elim": int(n_elim),
                "n_withdrew": int(n_withdrew),
                "has_elim": int(has_elim),
                "has_withdrew": int(has_withdrew),

                "U_week_rel_rci90_mean": float(U_week_rel_rci90_mean),

                "judge_gap": float(judge_gap),
                "judge_concentration": float(judge_concentration),
            })

    return pd.DataFrame(per_rows), pd.DataFrame(diag_rows)


# Main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Q1")
    ap.add_argument("--data", type=str, default=None, help="Path to data csv")
    ap.add_argument("--out", type=str, default=None, help="Output directory")
    args = ap.parse_args()

    csv_path = get_data_path(args.data)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(str(csv_path))

    per_week, diagnostics = solve_q1_three_rules_dynamic(df=df)

    print("A: Per contestant-week (head):")
    print(per_week.head())

    print("\nB: Week table (head):")
    print(diagnostics.head())

    out_root = ensure_dir(get_out_dir("Q1", args.out))
    tables_dir = ensure_dir(out_root / "tables")

    out1 = tables_dir / ("Q1_1_test.csv" if TEST_MODE else "Q1_1.csv")
    out2 = tables_dir / ("Q1_2_test.csv" if TEST_MODE else "Q1_2.csv")

    per_week.to_csv(str(out1), index=False)
    diagnostics.to_csv(str(out2), index=False)

    print("\nSaved outputs:")
    print(" -", str(out1))
    print(" -", str(out2))


