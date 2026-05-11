"""
Phase 9: Recency decay weighting.

Apply weight = exp(-days_ago / (365 * half_life_years)) to training samples.
days_ago is measured relative to the training cutoff date (= start of test year).

Test multiple half-lives. Best half-life should balance:
  - Too short (1y): throws away too much historical pattern data
  - Too long (10y): doesn't differentiate recent vs old enough
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
from sklearn.metrics import accuracy_score, brier_score_loss

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


def probs(lh, la, K=10):
    ph = poisson.pmf(np.arange(K+1), lh); pa = poisson.pmf(np.arange(K+1), la)
    g = np.outer(ph, pa)
    return float(np.tril(g,-1).sum()), float(np.diag(g).sum()), float(np.triu(g,1).sum())


def eval_set(test, hm, am):
    test = test.dropna(subset=FEATS).copy()
    X = test[FEATS].values
    p = np.array([probs(h, a) for h, a in zip(hm.predict(X), am.predict(X))])
    test["p_h"], test["p_d"], test["p_a"] = p[:,0], p[:,1], p[:,2]
    test["pred"] = np.where(test["p_h"] > np.maximum(test["p_d"], test["p_a"]), 1,
                    np.where(test["p_a"] > np.maximum(test["p_h"], test["p_d"]), -1, 0))
    y = test["result"].values
    return {
        "n": len(test),
        "acc": round(accuracy_score(y, test["pred"]), 4),
        "brier": round((brier_score_loss((y==1).astype(int), test["p_h"]) +
                        brier_score_loss((y==0).astype(int), test["p_d"]) +
                        brier_score_loss((y==-1).astype(int), test["p_a"])) / 3, 4),
    }


def run_scheme(df, test_pool, years, half_life: float | None) -> tuple[float, float]:
    """Returns (mean_acc, mean_brier). half_life=None for uniform weighting."""
    accs, briers = [], []
    for y in years:
        cutoff = pd.Timestamp(f"{y}-01-01")
        train = df[(df["date"] >= "2000-01-01") & (df["date"] < cutoff)]
        test = test_pool[test_pool["date"].dt.year == y]
        if len(train) < 100 or len(test) < 5:
            accs.append(np.nan); briers.append(np.nan); continue

        if half_life is None:
            sw = None
        else:
            days_ago = (cutoff - train["date"]).dt.days.values
            sw = np.exp(-days_ago / (365.0 * half_life))

        fit_kwargs = {"p__sample_weight": sw} if sw is not None else {}
        hm = make_pipe().fit(train[FEATS], train["home_score"], **fit_kwargs)
        am = make_pipe().fit(train[FEATS], train["away_score"], **fit_kwargs)
        r = eval_set(test, hm, am)
        accs.append(r["acc"]); briers.append(r["brier"])
    return float(np.nanmean(accs)), float(np.nanmean(briers)), accs


def main():
    conn = sqlite3.connect(DB_PATH)
    df = load(conn).dropna(subset=FEATS)
    big = ["fifa world cup", "uefa euro", "copa américa", "copa america",
           "african cup of nations", "afc asian cup"]
    test_pool = df[df["t_low"].isin(big) & (df["date"] >= "2014-01-01")]
    print(f"Test pool size: {len(test_pool)}")
    years = [2014, 2016, 2018, 2020, 2022, 2024]

    schemes = [
        ("v2.9 (no decay)",      None),
        ("v4a (HL=1 yr)",        1.0),
        ("v4b (HL=2 yr)",        2.0),
        ("v4c (HL=3 yr)",        3.0),
        ("v4d (HL=5 yr)",        5.0),
        ("v4e (HL=8 yr)",        8.0),
        ("v4f (HL=15 yr)",       15.0),
    ]

    print(f"\n{'scheme':25s}  " + "  ".join(f"{y:6d}" for y in years) + f"   {'mean':>6s}  {'brier':>7s}")
    print("-" * 100)
    results = []
    for label, hl in schemes:
        mean_acc, mean_b, accs = run_scheme(df, test_pool, years, hl)
        accs_str = "  ".join(f"{a:.3f}" if not np.isnan(a) else " --- " for a in accs)
        marker = "  ✓" if hl is None or label == "v2.9 (no decay)" else ""
        print(f"{label:25s}  {accs_str}   {mean_acc:.3f}   {mean_b:.4f}{marker}")
        results.append((label, hl, mean_acc, mean_b))

    # find best
    best = max(results, key=lambda r: r[2])
    baseline = results[0]
    print()
    print(f"Best: {best[0]}  (acc={best[2]:.3f})")
    print(f"Baseline v2.9: acc={baseline[2]:.3f}")
    print(f"Delta: {(best[2]-baseline[2])*100:+.2f}pp")

    conn.close()


if __name__ == "__main__":
    main()
