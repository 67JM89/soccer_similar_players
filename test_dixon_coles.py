"""
Phase 13: Dixon-Coles low-score correction on top of v3 Poisson model.

DC adjustment for low-score outcomes:
    τ(0,0) = 1 - λ_h × λ_a × ρ
    τ(1,0) = 1 + λ_a × ρ
    τ(0,1) = 1 + λ_h × ρ
    τ(1,1) = 1 - ρ
    otherwise = 1

ρ is estimated via grid search to maximize log-likelihood on training data.
Marginals of X_h and X_a remain Poisson — the τ correction shifts probability
mass between the 4 low-score cells only.
"""
import sys, io, sqlite3, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

DB_PATH = Path(__file__).parent / "data" / "soccer.db"
FEATS = ["elo_diff", "home_advantage", "form_gd_diff", "form_cs_diff"]


def make_pipe():
    return Pipeline([("s", StandardScaler()),
                     ("p", PoissonRegressor(alpha=0.01, max_iter=1000))])


def load(conn):
    df = pd.read_sql("""
      SELECT m.*, f.h_form_gd, f.h_form_cs, f.a_form_gd, f.a_form_cs
      FROM match_features m
      JOIN match_form f ON f.date=m.date AND f.home_team=m.home_team AND f.away_team=m.away_team
    """, conn)
    df["date"] = pd.to_datetime(df["date"])
    df["form_gd_diff"] = df["h_form_gd"] - df["a_form_gd"]
    df["form_cs_diff"] = df["h_form_cs"] - df["a_form_cs"]
    df["t_low"] = df["tournament"].str.lower()
    return df


def tau(a, b, lh, la, rho):
    """DC correction factor for cell (a, b)."""
    if a == 0 and b == 0: return 1 - lh * la * rho
    if a == 1 and b == 0: return 1 + la * rho
    if a == 0 and b == 1: return 1 + lh * rho
    if a == 1 and b == 1: return 1 - rho
    return 1.0


def estimate_rho(train: pd.DataFrame, hm, am) -> float:
    """Grid search for rho that maximizes log-likelihood on training data.
    Only the 4 low-score cells contribute to rho's likelihood gradient."""
    X = train[FEATS].values
    lh = hm.predict(X)
    la = am.predict(X)
    sh = train["home_score"].values.astype(int)
    sa = train["away_score"].values.astype(int)
    # mask of matches in the 4 low-score cells
    low = ((sh <= 1) & (sa <= 1))

    grid = np.linspace(-0.30, 0.10, 41)
    best_rho, best_ll = 0.0, -np.inf
    for rho in grid:
        # τ for each match (only relevant for low cells; otherwise τ=1)
        taus = np.ones(len(sh))
        taus[(sh == 0) & (sa == 0)] = 1 - lh[(sh == 0) & (sa == 0)] * la[(sh == 0) & (sa == 0)] * rho
        taus[(sh == 1) & (sa == 0)] = 1 + la[(sh == 1) & (sa == 0)] * rho
        taus[(sh == 0) & (sa == 1)] = 1 + lh[(sh == 0) & (sa == 1)] * rho
        taus[(sh == 1) & (sa == 1)] = 1 - rho
        # if any τ <= 0, this rho is invalid
        if (taus <= 0).any(): continue
        ll = np.sum(np.log(taus))   # only the τ part affects rho
        if ll > best_ll:
            best_ll, best_rho = ll, rho
    return float(best_rho)


def outcome_probs_dc(lh, la, rho, K=10):
    """Return (P(home), P(draw), P(away)) under Dixon-Coles."""
    ph = poisson.pmf(np.arange(K + 1), lh)
    pa = poisson.pmf(np.arange(K + 1), la)
    grid = np.outer(ph, pa)
    # Apply τ corrections
    grid[0, 0] *= (1 - lh * la * rho)
    grid[1, 0] *= (1 + la * rho)
    grid[0, 1] *= (1 + lh * rho)
    grid[1, 1] *= (1 - rho)
    return (float(np.tril(grid, -1).sum()),
            float(np.diag(grid).sum()),
            float(np.triu(grid, 1).sum()))


def outcome_probs_pois(lh, la, K=10):
    ph = poisson.pmf(np.arange(K + 1), lh)
    pa = poisson.pmf(np.arange(K + 1), la)
    g = np.outer(ph, pa)
    return (float(np.tril(g, -1).sum()),
            float(np.diag(g).sum()),
            float(np.triu(g, 1).sum()))


def eval_set(test: pd.DataFrame, hm, am, rho: float | None) -> dict:
    test = test.dropna(subset=FEATS).copy()
    X = test[FEATS].values
    lh = hm.predict(X); la = am.predict(X)
    prob_fn = (lambda h, a: outcome_probs_dc(h, a, rho)) if rho is not None else outcome_probs_pois
    p = np.array([prob_fn(h, a) for h, a in zip(lh, la)])
    test["p_h"] = p[:, 0]; test["p_d"] = p[:, 1]; test["p_a"] = p[:, 2]
    test["pred"] = np.where(test["p_h"] > np.maximum(test["p_d"], test["p_a"]), 1,
                    np.where(test["p_a"] > np.maximum(test["p_h"], test["p_d"]), -1, 0))
    y = test["result"].values
    actual_h = (y == 1).astype(int)
    actual_d = (y == 0).astype(int)
    actual_a = (y == -1).astype(int)
    brier = (brier_score_loss(actual_h, test["p_h"]) +
             brier_score_loss(actual_d, test["p_d"]) +
             brier_score_loss(actual_a, test["p_a"])) / 3.0
    P = test[["p_a", "p_d", "p_h"]].values
    cls = (y + 1).astype(int)
    ll = log_loss(cls, P, labels=[0, 1, 2])
    return {
        "n": len(test),
        "acc": round(accuracy_score(y, test["pred"]), 4),
        "brier": round(brier, 4),
        "log_loss": round(ll, 4),
        "draw_rate_actual":  round(actual_d.mean(), 3),
        "draw_rate_pred":    round(test["p_d"].mean(), 3),
    }


def main():
    conn = sqlite3.connect(DB_PATH)
    df = load(conn).dropna(subset=FEATS)

    big = ["fifa world cup", "uefa euro", "copa américa", "copa america",
           "african cup of nations", "afc asian cup"]
    test_pool = df[df["t_low"].isin(big) & (df["date"] >= "2014-01-01")]
    print(f"test pool: {len(test_pool)}")

    years = [2014, 2016, 2018, 2020, 2022, 2024]
    rows = []
    for y in years:
        train = df[(df["date"] >= "2000-01-01") & (df["date"] < f"{y}-01-01")]
        test = test_pool[test_pool["date"].dt.year == y]
        if len(train) < 100 or len(test) < 5:
            continue
        hm = make_pipe().fit(train[FEATS], train["home_score"])
        am = make_pipe().fit(train[FEATS], train["away_score"])

        # rho estimation on this fold's training data
        rho = estimate_rho(train, hm, am)

        r_pois = eval_set(test, hm, am, rho=None)
        r_dc   = eval_set(test, hm, am, rho=rho)
        rows.append({
            "year": y, "rho": round(rho, 4),
            "acc_pois": r_pois["acc"], "acc_dc": r_dc["acc"],
            "brier_pois": r_pois["brier"], "brier_dc": r_dc["brier"],
            "ll_pois": r_pois["log_loss"], "ll_dc": r_dc["log_loss"],
            "draw_actual": r_pois["draw_rate_actual"],
            "draw_pred_pois": r_pois["draw_rate_pred"],
            "draw_pred_dc": r_dc["draw_rate_pred"],
        })

    R = pd.DataFrame(rows)
    print()
    print(R.to_string(index=False))
    print()
    print("Means:")
    print(f"  Poisson:       acc={R['acc_pois'].mean():.4f}  brier={R['brier_pois'].mean():.4f}  log_loss={R['ll_pois'].mean():.4f}")
    print(f"  Dixon-Coles:   acc={R['acc_dc'].mean():.4f}  brier={R['brier_dc'].mean():.4f}  log_loss={R['ll_dc'].mean():.4f}")
    print(f"  Δ accuracy:    {(R['acc_dc'].mean() - R['acc_pois'].mean()) * 100:+.2f}pp")
    print(f"  Δ brier:       {(R['brier_dc'].mean() - R['brier_pois'].mean()):+.4f}")
    print(f"  Δ log_loss:    {(R['ll_dc'].mean() - R['ll_pois'].mean()):+.4f}")
    print(f"  Draw rate actual:  {R['draw_actual'].mean():.3f}")
    print(f"  Draw rate Pois:    {R['draw_pred_pois'].mean():.3f}")
    print(f"  Draw rate DC:      {R['draw_pred_dc'].mean():.3f}")
    print(f"  Estimated ρ mean:  {R['rho'].mean():.4f}")

    conn.close()


if __name__ == "__main__":
    main()
