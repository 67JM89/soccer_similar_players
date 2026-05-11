"""
Phase 14: XGBoost regression for goal prediction (Poisson objective).

Compare:
  - v3 (Poisson regression, scaled features)  ← current best
  - v4a XGBoost with same features
  - v4b XGBoost with extended features
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
import xgboost as xgb

DB_PATH = Path(__file__).parent / "data" / "soccer.db"


def make_poisson_pipe():
    return Pipeline([("s", StandardScaler()),
                     ("p", PoissonRegressor(alpha=0.01, max_iter=1000))])


def make_xgb(n_estimators=200, max_depth=4, lr=0.05):
    return xgb.XGBRegressor(
        objective="count:poisson",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def load(conn):
    df = pd.read_sql("""
      SELECT m.*,
             f.h_form_ppg, f.h_form_gd, f.h_form_cs,
             f.a_form_ppg, f.a_form_gd, f.a_form_cs
      FROM match_features m
      JOIN match_form f
        ON f.date=m.date AND f.home_team=m.home_team AND f.away_team=m.away_team
    """, conn)
    df["date"] = pd.to_datetime(df["date"])
    df["form_ppg_diff"] = df["h_form_ppg"] - df["a_form_ppg"]
    df["form_gd_diff"]  = df["h_form_gd"]  - df["a_form_gd"]
    df["form_cs_diff"]  = df["h_form_cs"]  - df["a_form_cs"]
    df["t_low"] = df["tournament"].str.lower()
    return df


def probs(lh, la, K=10):
    ph = poisson.pmf(np.arange(K+1), lh); pa = poisson.pmf(np.arange(K+1), la)
    g = np.outer(ph, pa)
    return float(np.tril(g,-1).sum()), float(np.diag(g).sum()), float(np.triu(g,1).sum())


def eval_set(test, feats, hm, am, predict_fn):
    test = test.dropna(subset=feats).copy()
    X = test[feats].values
    lh = predict_fn(hm, X); la = predict_fn(am, X)
    p = np.array([probs(h, a) for h, a in zip(lh, la)])
    test["p_h"] = p[:,0]; test["p_d"] = p[:,1]; test["p_a"] = p[:,2]
    test["pred"] = np.where(test["p_h"] > np.maximum(test["p_d"], test["p_a"]), 1,
                    np.where(test["p_a"] > np.maximum(test["p_h"], test["p_d"]), -1, 0))
    y = test["result"].values
    actual = {1: (y == 1).astype(int), 0: (y == 0).astype(int), -1: (y == -1).astype(int)}
    brier = (brier_score_loss(actual[1],  test["p_h"]) +
             brier_score_loss(actual[0],  test["p_d"]) +
             brier_score_loss(actual[-1], test["p_a"])) / 3.0
    P = test[["p_a", "p_d", "p_h"]].values
    cls = (y + 1).astype(int)
    ll = log_loss(cls, P, labels=[0, 1, 2])
    return {"acc": round(accuracy_score(y, test["pred"]), 4),
            "brier": round(brier, 4), "log_loss": round(ll, 4)}


def run_variant(df, test_pool, years, feats, model_fn, predict_fn, label):
    accs, briers, lls = [], [], []
    for y in years:
        train = df[(df["date"] >= "2000-01-01") & (df["date"] < f"{y}-01-01")].dropna(subset=feats)
        test = test_pool[test_pool["date"].dt.year == y]
        if len(train) < 100 or len(test) < 5:
            accs.append(np.nan); briers.append(np.nan); lls.append(np.nan); continue
        hm = model_fn(); hm.fit(train[feats].values, train["home_score"].values)
        am = model_fn(); am.fit(train[feats].values, train["away_score"].values)
        r = eval_set(test, feats, hm, am, predict_fn)
        accs.append(r["acc"]); briers.append(r["brier"]); lls.append(r["log_loss"])
    return {
        "label": label, "mean_acc": float(np.nanmean(accs)),
        "mean_brier": float(np.nanmean(briers)), "mean_ll": float(np.nanmean(lls)),
        "accs": accs,
    }


def main():
    conn = sqlite3.connect(DB_PATH)
    df = load(conn)
    big = ["fifa world cup", "uefa euro", "copa américa", "copa america",
           "african cup of nations", "afc asian cup"]
    test_pool = df[df["t_low"].isin(big) & (df["date"] >= "2014-01-01")]
    print(f"test pool: {len(test_pool)}")

    years = [2014, 2016, 2018, 2020, 2022, 2024]
    v3_feats = ["elo_diff", "home_advantage", "form_gd_diff", "form_cs_diff"]
    ext_feats = v3_feats + [
        "elo_home_pre", "elo_away_pre", "form_ppg_diff",
        "h_form_gd", "h_form_cs", "a_form_gd", "a_form_cs",
    ]

    variants = [
        ("v3 Poisson (4 feats)",         v3_feats,  make_poisson_pipe,
            lambda m, X: m.predict(X)),
        ("v4a XGBoost (4 feats)",        v3_feats,  make_xgb,
            lambda m, X: m.predict(X)),
        ("v4b XGBoost (11 feats)",       ext_feats, make_xgb,
            lambda m, X: m.predict(X)),
        ("v4c XGBoost shallow (4)",      v3_feats,  lambda: make_xgb(n_estimators=100, max_depth=3),
            lambda m, X: m.predict(X)),
        ("v4d XGBoost deep (11)",        ext_feats, lambda: make_xgb(n_estimators=400, max_depth=6, lr=0.03),
            lambda m, X: m.predict(X)),
    ]

    print(f"\n{'variant':30s}  " + "  ".join(f"{y:>6d}" for y in years) + f"   {'mean':>6s}  {'brier':>7s}  {'ll':>6s}")
    print("-" * 110)
    results = []
    for label, feats, mf, pf in variants:
        r = run_variant(df, test_pool, years, feats, mf, pf, label)
        accs_str = "  ".join(f"{a:.3f}" if not np.isnan(a) else "  --- " for a in r["accs"])
        print(f"{label:30s}  {accs_str}   {r['mean_acc']:.3f}   {r['mean_brier']:.4f}   {r['mean_ll']:.4f}")
        results.append(r)

    best = max(results, key=lambda r: r["mean_acc"])
    baseline = results[0]
    print(f"\nBest by acc: {best['label']} (acc={best['mean_acc']:.4f})")
    print(f"v3 baseline: acc={baseline['mean_acc']:.4f}")
    print(f"Δ acc:       {(best['mean_acc'] - baseline['mean_acc']) * 100:+.2f}pp")

    conn.close()


if __name__ == "__main__":
    main()
