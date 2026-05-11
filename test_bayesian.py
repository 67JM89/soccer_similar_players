"""
Phase 15: Bayesian Hierarchical model (MAP estimation via scipy).

Model:
    log λ_h = c + h_adv*(1-neutral) + atk[h] - def[a]
    log λ_a = c + atk[a] - def[h]
    home_score ~ Poisson(λ_h)
    away_score ~ Poisson(λ_a)
  + Dixon-Coles τ correction for low scores

Priors (shrinkage toward zero):
    atk[t] ~ Normal(0, σ_team)
    def[t] ~ Normal(0, σ_team)
    h_adv  ~ Normal(0.3, 0.3)
    rho    ~ Normal(0, 0.1)

Use MAP estimation via scipy L-BFGS for speed.
"""
import sys, io, sqlite3, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import gammaln
from scipy.stats import poisson
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

DB_PATH = Path(__file__).parent / "data" / "soccer.db"


# --------------------------------------------------------------- model
def neg_log_post(params: np.ndarray, h_idx, a_idx, neutral, hs, as_,
                 elo_diff_std,           # NEW: standardized ELO diff per match
                 T: int, sigma_team=0.5, sigma_h=0.3, sigma_rho=0.1, sigma_beta=1.0):
    atk = params[:T]
    de_ = params[T:2 * T]
    c     = params[2 * T]
    h_adv = params[2 * T + 1]
    rho   = params[2 * T + 2]
    beta  = params[2 * T + 3]    # NEW: coefficient on ELO diff

    log_lh = c + h_adv * (1 - neutral) + beta * elo_diff_std + atk[h_idx] - de_[a_idx]
    log_la = c                          - beta * elo_diff_std + atk[a_idx] - de_[h_idx]
    # clip to avoid overflow
    log_lh = np.clip(log_lh, -3, 3)
    log_la = np.clip(log_la, -3, 3)
    lh = np.exp(log_lh)
    la = np.exp(log_la)

    # Poisson log-likelihood
    ll_h = hs * log_lh - lh - gammaln(hs + 1)
    ll_a = as_ * log_la - la - gammaln(as_ + 1)
    ll = float(np.sum(ll_h + ll_a))

    # Dixon-Coles τ on low-score cells (vectorized)
    m00 = (hs == 0) & (as_ == 0)
    m10 = (hs == 1) & (as_ == 0)
    m01 = (hs == 0) & (as_ == 1)
    m11 = (hs == 1) & (as_ == 1)

    t00 = 1 - lh[m00] * la[m00] * rho
    t10 = 1 + la[m10] * rho
    t01 = 1 + lh[m01] * rho
    t11 = 1 - rho

    eps = 1e-8
    # add ln(tau) — guard against non-positive
    ll_tau = (np.sum(np.log(np.clip(t00, eps, None))) +
              np.sum(np.log(np.clip(t10, eps, None))) +
              np.sum(np.log(np.clip(t01, eps, None))) +
              float(np.sum(m11) * np.log(max(t11, eps))))
    ll += ll_tau

    # Priors (Gaussian, log-prior = -0.5 * (x - μ)^2 / σ^2)
    log_prior = (-0.5 * np.sum(atk * atk) / sigma_team**2
                 -0.5 * np.sum(de_ * de_) / sigma_team**2
                 -0.5 * (h_adv - 0.3) ** 2 / sigma_h ** 2
                 -0.5 * c * c / 1.0**2
                 -0.5 * (rho ** 2) / sigma_rho ** 2
                 -0.5 * (beta ** 2) / sigma_beta ** 2)

    return -(ll + log_prior)


def fit_map(train: pd.DataFrame, teams: dict[str, int]):
    T = len(teams)
    h_idx = train["home_team"].map(teams).values
    a_idx = train["away_team"].map(teams).values
    valid = (~np.isnan(h_idx.astype(float))) & (~np.isnan(a_idx.astype(float)))
    train = train[valid]
    h_idx = train["home_team"].map(teams).values.astype(int)
    a_idx = train["away_team"].map(teams).values.astype(int)
    neutral = train["neutral"].values.astype(float)
    hs = train["home_score"].values.astype(int)
    as_ = train["away_score"].values.astype(int)

    # Standardize elo_diff to ~unit variance for stable optimization
    ed_raw = train["elo_diff"].values
    ed_mean, ed_std = float(ed_raw.mean()), float(ed_raw.std() + 1e-9)
    elo_diff_std = (ed_raw - ed_mean) / ed_std

    # 2T + 4 params (was +3, now +beta)
    x0 = np.zeros(2 * T + 4)
    x0[2 * T] = np.log(max(hs.mean(), 0.5))   # c
    x0[2 * T + 1] = 0.3                        # h_adv
    x0[2 * T + 2] = -0.05                      # rho
    x0[2 * T + 3] = 0.4                        # beta (typical scaled-ELO effect)

    res = minimize(
        neg_log_post, x0,
        args=(h_idx, a_idx, neutral, hs, as_, elo_diff_std, T),
        method="L-BFGS-B",
        options={"maxiter": 400, "disp": False},
    )
    return res, (ed_mean, ed_std)


def predict_match(params, t_a: str, t_b: str, neutral: int, elo_diff_std: float,
                  teams: dict[str, int], K: int = 8):
    if t_a not in teams or t_b not in teams:
        return (np.nan, np.nan, np.nan)
    T = len(teams)
    atk = params[:T]; de_ = params[T:2*T]
    c, h_adv, rho, beta = params[2*T], params[2*T+1], params[2*T+2], params[2*T+3]
    log_lh = c + h_adv * (1 - neutral) + beta * elo_diff_std + atk[teams[t_a]] - de_[teams[t_b]]
    log_la = c                          - beta * elo_diff_std + atk[teams[t_b]] - de_[teams[t_a]]
    lh = float(np.exp(np.clip(log_lh, -3, 3)))
    la = float(np.exp(np.clip(log_la, -3, 3)))
    ph = poisson.pmf(np.arange(K + 1), lh)
    pa = poisson.pmf(np.arange(K + 1), la)
    g = np.outer(ph, pa)
    g[0, 0] *= (1 - lh * la * rho)
    g[1, 0] *= (1 + la * rho)
    g[0, 1] *= (1 + lh * rho)
    g[1, 1] *= (1 - rho)
    return (float(np.tril(g, -1).sum()),
            float(np.diag(g).sum()),
            float(np.triu(g, 1).sum()))


# --------------------------------------------------------------- evaluation
def evaluate(test, params, teams, ed_norm):
    ed_mean, ed_std = ed_norm
    preds = []
    valid_idx = []
    for i, r in test.iterrows():
        ed = (r["elo_diff"] - ed_mean) / ed_std
        p_a, p_d, p_h = predict_match(params, r["home_team"], r["away_team"],
                                      int(r["neutral"]), ed, teams)
        if any(np.isnan([p_a, p_d, p_h])):
            continue
        preds.append((p_a, p_d, p_h))
        valid_idx.append(i)
    test = test.loc[valid_idx]
    P = np.array(preds)
    test["p_h"] = P[:, 0]; test["p_d"] = P[:, 1]; test["p_a"] = P[:, 2]
    test["pred"] = np.where(test["p_h"] > np.maximum(test["p_d"], test["p_a"]), 1,
                    np.where(test["p_a"] > np.maximum(test["p_h"], test["p_d"]), -1, 0))
    y = test["result"].values
    actual = {1: (y == 1).astype(int), 0: (y == 0).astype(int), -1: (y == -1).astype(int)}
    brier = (brier_score_loss(actual[1], test["p_h"]) +
             brier_score_loss(actual[0], test["p_d"]) +
             brier_score_loss(actual[-1], test["p_a"])) / 3.0
    Pmat = test[["p_a", "p_d", "p_h"]].values
    cls = (y + 1).astype(int)
    ll = log_loss(cls, Pmat, labels=[0, 1, 2])
    return {
        "n": len(test),
        "acc": round(accuracy_score(y, test["pred"]), 4),
        "brier": round(brier, 4),
        "log_loss": round(ll, 4),
    }


def main():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM match_features", conn)
    df["date"] = pd.to_datetime(df["date"])
    df["t_low"] = df["tournament"].str.lower()
    df = df.dropna(subset=["home_score", "away_score"])

    big = ["fifa world cup", "uefa euro", "copa américa", "copa america",
           "african cup of nations", "afc asian cup"]
    test_pool = df[df["t_low"].isin(big) & (df["date"] >= "2014-01-01")]
    print(f"test pool: {len(test_pool)}")

    years = [2014, 2016, 2018, 2020, 2022, 2024]
    rows = []
    for y in years:
        train = df[(df["date"] >= "2000-01-01") & (df["date"] < f"{y}-01-01")]
        test = test_pool[test_pool["date"].dt.year == y]
        if len(test) < 5:
            continue
        # Build team index from training matches that have at least 5 games
        team_counts = pd.concat([train["home_team"], train["away_team"]]).value_counts()
        active_teams = team_counts[team_counts >= 5].index.tolist()
        teams = {t: i for i, t in enumerate(active_teams)}
        train = train[train["home_team"].isin(teams) & train["away_team"].isin(teams)]

        print(f"\n>> Year {y}  train={len(train):,}  teams={len(teams)}  test={len(test)}")
        t0 = time.time()
        res, ed_norm = fit_map(train, teams)
        elapsed = time.time() - t0
        beta = res.x[-1]
        print(f"   fit MAP in {elapsed:.1f}s,  converged={res.success},  -ll={res.fun:.1f},  β(elo)={beta:.3f}")

        r = evaluate(test.copy(), res.x, teams, ed_norm)
        print(f"   eval: {r}")
        rows.append({"year": y, "n_train": len(train), "n_teams": len(teams),
                     "fit_s": round(elapsed, 1),
                     "acc": r["acc"], "brier": r["brier"], "log_loss": r["log_loss"], "n_test": r["n"]})

    R = pd.DataFrame(rows)
    print()
    print(R.to_string(index=False))
    print()
    print(f"Bayesian (MAP) means:  acc={R['acc'].mean():.4f}  brier={R['brier'].mean():.4f}  log_loss={R['log_loss'].mean():.4f}")
    print()
    print("v3 baseline (for comparison):  acc=0.5426  brier=0.1963  log_loss=0.9880")
    print(f"Δ accuracy:    {(R['acc'].mean() - 0.5426) * 100:+.2f}pp")
    print(f"Δ brier:       {(R['brier'].mean() - 0.1963):+.4f}")
    print(f"Δ log_loss:    {(R['log_loss'].mean() - 0.9880):+.4f}")

    conn.close()


if __name__ == "__main__":
    main()
