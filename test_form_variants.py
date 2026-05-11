"""Test single-form-metric variants vs v1 baseline."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import sqlite3
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

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

def split_wc(df, end, year):
    tr = df[(df["date"] >= "2000-01-01") & (df["date"] < end)]
    te = df[(df["tournament"].str.lower() == "fifa world cup") & (df["date"].dt.year == year)]
    return tr, te

def probs(lh, la, K=10):
    ph = poisson.pmf(np.arange(K+1), lh); pa = poisson.pmf(np.arange(K+1), la)
    g = np.outer(ph, pa)
    return float(np.tril(g,-1).sum()), float(np.diag(g).sum()), float(np.triu(g,1).sum())

def evaluate(test, feats, hm, am):
    test = test.dropna(subset=feats).copy()
    X = test[feats].values
    lh = hm.predict(X); la = am.predict(X)
    p = np.array([probs(h, a) for h, a in zip(lh, la)])
    test["p_h"] = p[:,0]; test["p_d"] = p[:,1]; test["p_a"] = p[:,2]
    test["pred"] = np.where(test["p_h"] > np.maximum(test["p_d"], test["p_a"]), 1,
                    np.where(test["p_a"] > np.maximum(test["p_h"], test["p_d"]), -1, 0))
    y = test["result"].values
    return {
        "n": len(test),
        "accuracy": round(accuracy_score(y, test["pred"]), 4),
        "brier": round((brier_score_loss((y==1).astype(int), test["p_h"]) +
                        brier_score_loss((y==0).astype(int), test["p_d"]) +
                        brier_score_loss((y==-1).astype(int), test["p_a"])) / 3, 4),
    }

def run(df, feats, label):
    tr22, te22 = split_wc(df, "2022-11-01", 2022)
    tr18, te18 = split_wc(df, "2018-06-01", 2018)
    tr22 = tr22.dropna(subset=feats); tr18 = tr18.dropna(subset=feats)
    hm22 = make_pipe().fit(tr22[feats], tr22["home_score"])
    am22 = make_pipe().fit(tr22[feats], tr22["away_score"])
    hm18 = make_pipe().fit(tr18[feats], tr18["home_score"])
    am18 = make_pipe().fit(tr18[feats], tr18["away_score"])
    r22 = evaluate(te22, feats, hm22, am22)
    r18 = evaluate(te18, feats, hm18, am18)
    print(f"  {label:30s} 2022: acc={r22['accuracy']:.3f} brier={r22['brier']:.3f} | 2018: acc={r18['accuracy']:.3f} brier={r18['brier']:.3f}")
    return r22, r18


def main():
    conn = sqlite3.connect(DB_PATH)
    df = load(conn)
    base = ["elo_diff", "home_advantage"]
    print("Variants:")
    run(df, base, "v1: ELO only")
    run(df, base + ["form_ppg_diff"], "v2.6: + form_ppg_diff")
    run(df, base + ["form_gd_diff"],  "v2.5: + form_gd_diff")
    run(df, base + ["form_cs_diff"],  "v2.7: + form_cs_diff")
    run(df, base + ["form_ppg_diff", "form_cs_diff"], "v2.8: + ppg + cs")
    run(df, base + ["form_gd_diff",  "form_cs_diff"], "v2.9: + gd  + cs")
    conn.close()


if __name__ == "__main__":
    main()
