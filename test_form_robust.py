"""Test form variants on a larger pooled test set across multiple major tournaments."""
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

def make_pipe():
    return Pipeline([("s", StandardScaler()),
                     ("p", PoissonRegressor(alpha=0.01, max_iter=1000))])

def load(conn):
    df = pd.read_sql("""
      SELECT m.*, f.h_form_ppg, f.h_form_gd, f.h_form_cs,
             f.a_form_ppg, f.a_form_gd, f.a_form_cs
      FROM match_features m
      JOIN match_form f ON f.date=m.date AND f.home_team=m.home_team AND f.away_team=m.away_team
    """, conn)
    df["date"] = pd.to_datetime(df["date"])
    df["form_ppg_diff"] = df["h_form_ppg"] - df["a_form_ppg"]
    df["form_gd_diff"]  = df["h_form_gd"]  - df["a_form_gd"]
    df["form_cs_diff"]  = df["h_form_cs"]  - df["a_form_cs"]
    return df

def probs(lh, la, K=10):
    ph = poisson.pmf(np.arange(K+1), lh); pa = poisson.pmf(np.arange(K+1), la)
    g = np.outer(ph, pa)
    return float(np.tril(g,-1).sum()), float(np.diag(g).sum()), float(np.triu(g,1).sum())

def eval_set(test, feats, hm, am):
    test = test.dropna(subset=feats).copy()
    X = test[feats].values
    lh, la = hm.predict(X), am.predict(X)
    p = np.array([probs(h, a) for h, a in zip(lh, la)])
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

def main():
    conn = sqlite3.connect(DB_PATH)
    df = load(conn)

    # Build a LARGER pooled test set: every major tournament knockout/group game
    # since 2014, EXCLUDING the train period
    big_tournaments = [
        "fifa world cup", "uefa euro", "copa américa", "copa america",
        "africa cup of nations", "afc asian cup",
    ]
    df["t_low"] = df["tournament"].str.lower()

    # Test set: 2014-2024 major tournament matches (~600+ games)
    test_mask = (df["t_low"].isin(big_tournaments)) & (df["date"] >= "2014-01-01")
    test_all = df[test_mask].copy()
    print(f"Pooled test set size: {len(test_all)}")

    variants = {
        "v1: ELO only":            ["elo_diff", "home_advantage"],
        "v2.6: +ppg":              ["elo_diff", "home_advantage", "form_ppg_diff"],
        "v2.5: +gd":               ["elo_diff", "home_advantage", "form_gd_diff"],
        "v2.7: +cs":               ["elo_diff", "home_advantage", "form_cs_diff"],
        "v2.9: +gd+cs":            ["elo_diff", "home_advantage", "form_gd_diff", "form_cs_diff"],
        "v2.full: +ppg+gd+cs":     ["elo_diff", "home_advantage", "form_ppg_diff", "form_gd_diff", "form_cs_diff"],
    }

    # Walk-forward: train on data BEFORE each test year
    print("\nWalk-forward by year:")
    print(f"{'variant':25s}  {'2014':6s}  {'2016':6s}  {'2018':6s}  {'2020':6s}  {'2022':6s}  {'2024':6s}  {'mean':6s}")
    print("-" * 90)

    test_years = [2014, 2016, 2018, 2020, 2022, 2024]
    for label, feats in variants.items():
        accs = []
        for y in test_years:
            train = df[(df["date"] >= "2000-01-01") & (df["date"] < f"{y}-01-01")].dropna(subset=feats)
            test = test_all[test_all["date"].dt.year == y]
            if len(train) < 100 or len(test) < 5:
                accs.append(np.nan); continue
            hm = make_pipe().fit(train[feats], train["home_score"])
            am = make_pipe().fit(train[feats], train["away_score"])
            r = eval_set(test, feats, hm, am)
            accs.append(r["acc"])
        mean = np.nanmean(accs)
        accs_str = "  ".join(f"{a:.3f}" if not np.isnan(a) else " --- " for a in accs)
        print(f"{label:25s}  {accs_str}  {mean:.3f}")

    conn.close()


if __name__ == "__main__":
    main()
