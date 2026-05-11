"""
Phase 7b: Train Poisson model with ELO + Recent Form features.
Backtests vs v1 (ELO only) on 2018 and 2022 World Cups.

Features (v2):
  elo_diff           : (home_elo + home_advantage) - away_elo
  home_advantage     : 0 or 100
  form_ppg_diff      : home form_ppg - away form_ppg     (last 10 matches each)
  form_gd_diff       : home form_gd  - away form_gd
  form_cs_diff       : home form_cs  - away form_cs
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

DB_PATH = Path(__file__).parent / "data" / "soccer.db"

V1_FEATS = ["elo_diff", "home_advantage"]
V2_FEATS = ["elo_diff", "home_advantage",
            "form_ppg_diff", "form_gd_diff", "form_cs_diff"]


def make_pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        ("poisson", PoissonRegressor(alpha=0.01, max_iter=1000)),
    ])


def load_features(conn) -> pd.DataFrame:
    q = """
    SELECT m.*,
           f.h_form_ppg, f.h_form_gd, f.h_form_cs,
           f.a_form_ppg, f.a_form_gd, f.a_form_cs
    FROM match_features m
    JOIN match_form f
      ON f.date = m.date AND f.home_team = m.home_team AND f.away_team = m.away_team
    """
    df = pd.read_sql(q, conn)
    df["date"] = pd.to_datetime(df["date"])
    df["form_ppg_diff"] = df["h_form_ppg"] - df["a_form_ppg"]
    df["form_gd_diff"]  = df["h_form_gd"]  - df["a_form_gd"]
    df["form_cs_diff"]  = df["h_form_cs"]  - df["a_form_cs"]
    return df


def split_wc(df, train_end: str, year: int):
    train = df[(df["date"] >= "2000-01-01") & (df["date"] < train_end)].copy()
    test = df[(df["tournament"].str.lower() == "fifa world cup")
              & (df["date"].dt.year == year)].copy()
    return train, test


def outcome_probs(lh, la, K=10):
    ph = poisson.pmf(np.arange(K + 1), lh)
    pa = poisson.pmf(np.arange(K + 1), la)
    g = np.outer(ph, pa)
    return float(np.tril(g, -1).sum()), float(np.diag(g).sum()), float(np.triu(g, 1).sum())


def evaluate(test: pd.DataFrame, feats: list[str], hm, am) -> dict:
    test = test.dropna(subset=feats).copy()
    X = test[feats].values
    lam_h = hm.predict(X)
    lam_a = am.predict(X)
    probs = np.array([outcome_probs(h, a) for h, a in zip(lam_h, lam_a)])
    test["p_home"] = probs[:, 0]; test["p_draw"] = probs[:, 1]; test["p_away"] = probs[:, 2]
    test["pred"] = np.where(test["p_home"] > np.maximum(test["p_draw"], test["p_away"]), 1,
                    np.where(test["p_away"] > np.maximum(test["p_home"], test["p_draw"]), -1, 0))

    y = test["result"].values
    acc = accuracy_score(y, test["pred"])
    actual = {1: (y == 1).astype(int), 0: (y == 0).astype(int), -1: (y == -1).astype(int)}
    brier = (
        brier_score_loss(actual[1],  test["p_home"]) +
        brier_score_loss(actual[0],  test["p_draw"]) +
        brier_score_loss(actual[-1], test["p_away"])
    ) / 3.0
    P = test[["p_away", "p_draw", "p_home"]].values
    cls = (y + 1).astype(int)
    ll = log_loss(cls, P, labels=[0, 1, 2])
    return {"n": len(test), "accuracy": round(acc, 4),
            "brier": round(brier, 4), "log_loss": round(ll, 4)}


def train_and_eval(df, feats, label):
    print(f"\n--- {label}  features={feats}")
    # Backtest 2022
    tr, te = split_wc(df, "2022-11-01", 2022)
    tr = tr.dropna(subset=feats)
    print(f"  2022 WC | train={len(tr):,} test={len(te)}")
    Xh, yh, ya = tr[feats].values, tr["home_score"].values, tr["away_score"].values
    hm = make_pipeline().fit(Xh, yh)
    am = make_pipeline().fit(Xh, ya)
    m22 = evaluate(te, feats, hm, am)
    print(f"  2022 WC | {m22}")

    # Backtest 2018
    tr18, te18 = split_wc(df, "2018-06-01", 2018)
    tr18 = tr18.dropna(subset=feats)
    Xh18 = tr18[feats].values
    hm18 = make_pipeline().fit(Xh18, tr18["home_score"].values)
    am18 = make_pipeline().fit(Xh18, tr18["away_score"].values)
    m18 = evaluate(te18, feats, hm18, am18)
    print(f"  2018 WC | {m18}")

    return {"2022": m22, "2018": m18}


def fit_final(df, feats):
    full = df[df["date"] >= "2000-01-01"].dropna(subset=feats)
    Xh = full[feats].values
    hm = make_pipeline().fit(Xh, full["home_score"].values)
    am = make_pipeline().fit(Xh, full["away_score"].values)
    return hm, am, len(full)


def main():
    conn = sqlite3.connect(DB_PATH)
    df = load_features(conn)
    print(f"Loaded {len(df):,} matches with form features")
    print(f"Rows with valid form (both teams): {len(df.dropna(subset=V2_FEATS)):,}")

    print("\n" + "=" * 70)
    print("BACKTEST COMPARISON  v1 (ELO only)  vs  v2 (ELO + Form)")
    print("=" * 70)
    r_v1 = train_and_eval(df, V1_FEATS, "v1 (ELO only)")
    r_v2 = train_and_eval(df, V2_FEATS, "v2 (ELO + Form)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  2022 WC accuracy:  v1={r_v1['2022']['accuracy']:.3f}  -> v2={r_v2['2022']['accuracy']:.3f}  "
          f"(Δ={(r_v2['2022']['accuracy']-r_v1['2022']['accuracy']):+.3f})")
    print(f"  2018 WC accuracy:  v1={r_v1['2018']['accuracy']:.3f}  -> v2={r_v2['2018']['accuracy']:.3f}  "
          f"(Δ={(r_v2['2018']['accuracy']-r_v1['2018']['accuracy']):+.3f})")
    print(f"  2022 Brier:        v1={r_v1['2022']['brier']:.3f}    -> v2={r_v2['2022']['brier']:.3f}")
    print(f"  2018 Brier:        v1={r_v1['2018']['brier']:.3f}    -> v2={r_v2['2018']['brier']:.3f}")
    print(f"  2022 Log-loss:     v1={r_v1['2022']['log_loss']:.3f}    -> v2={r_v2['2022']['log_loss']:.3f}")
    print(f"  2018 Log-loss:     v1={r_v1['2018']['log_loss']:.3f}    -> v2={r_v2['2018']['log_loss']:.3f}")

    # Train final v2 on everything for use in predictions
    hm, am, n = fit_final(df, V2_FEATS)
    print(f"\nFinal v2 model trained on {n:,} matches")
    h_pois = hm.named_steps["poisson"]; a_pois = am.named_steps["poisson"]
    print(f"  home coefs: {dict(zip(V2_FEATS, h_pois.coef_.round(4)))}")
    print(f"  away coefs: {dict(zip(V2_FEATS, a_pois.coef_.round(4)))}")

    conn.close()


if __name__ == "__main__":
    main()
