"""
Phase 8: Test match-importance weighting on top of v2.9 features.

Goal: down-weight friendlies (noisy) and weight major tournament matches more.
Test multiple weighting schemes via 6-year walk-forward backtest.
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
      SELECT m.*, f.h_form_ppg, f.h_form_gd, f.h_form_cs,
             f.a_form_ppg, f.a_form_gd, f.a_form_cs
      FROM match_features m
      JOIN match_form f ON f.date=m.date AND f.home_team=m.home_team AND f.away_team=m.away_team
    """, conn)
    df["date"] = pd.to_datetime(df["date"])
    df["form_gd_diff"] = df["h_form_gd"] - df["a_form_gd"]
    df["form_cs_diff"] = df["h_form_cs"] - df["a_form_cs"]
    df["t_low"] = df["tournament"].str.lower()
    return df


# ---- weighting schemes -------------------------------------------------
TIER_A = {"fifa world cup", "uefa euro", "copa américa", "copa america",
          "african cup of nations", "afc asian cup", "concacaf gold cup",
          "uefa nations league", "concacaf nations league", "gold cup"}


def w_uniform(t: str) -> float:
    return 1.0


def w_3tier(t: str) -> float:
    """Major: 1.0, qualifiers: 0.5, friendly: 0.2, other: 0.5"""
    if t in TIER_A: return 1.0
    if "qualif" in t: return 0.5
    if "friendly" in t: return 0.2
    return 0.5


def w_4tier(t: str) -> float:
    """Major: 1.0, qualifiers: 0.7, other tournament: 0.4, friendly: 0.2"""
    if t in TIER_A: return 1.0
    if "qualif" in t: return 0.7
    if "friendly" in t: return 0.2
    return 0.4


def w_aggressive(t: str) -> float:
    """Heavily favor majors"""
    if t in TIER_A: return 1.0
    if "qualif" in t: return 0.4
    if "friendly" in t: return 0.1
    return 0.3


def w_friendly_only(t: str) -> float:
    """Just down-weight friendlies"""
    return 0.2 if "friendly" in t else 1.0


SCHEMES = {
    "v2.9 (uniform)":     w_uniform,
    "v3a (friendly→0.2)": w_friendly_only,
    "v3b (3-tier)":       w_3tier,
    "v3c (4-tier)":       w_4tier,
    "v3d (aggressive)":   w_aggressive,
}


# ---- evaluation --------------------------------------------------------
def probs(lh, la, K=10):
    ph = poisson.pmf(np.arange(K+1), lh); pa = poisson.pmf(np.arange(K+1), la)
    g = np.outer(ph, pa)
    return float(np.tril(g,-1).sum()), float(np.diag(g).sum()), float(np.triu(g,1).sum())


def eval_set(test, hm, am):
    test = test.dropna(subset=FEATS).copy()
    X = test[FEATS].values
    p = np.array([probs(h, a) for h, a in zip(hm.predict(X), am.predict(X))])
    test["p_h"] = p[:,0]; test["p_d"] = p[:,1]; test["p_a"] = p[:,2]
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
    df = load(conn).dropna(subset=FEATS)

    big = ["fifa world cup", "uefa euro", "copa américa", "copa america",
           "african cup of nations", "afc asian cup"]
    test_pool = df[df["t_low"].isin(big) & (df["date"] >= "2014-01-01")]
    print(f"Test pool size: {len(test_pool)}")

    years = [2014, 2016, 2018, 2020, 2022, 2024]

    print(f"\n{'scheme':25s}  " + "  ".join(f"{y:6d}" for y in years) + f"   {'mean':>6s}  {'brier':>6s}")
    print("-" * 100)

    for label, wf in SCHEMES.items():
        accs, briers = [], []
        for y in years:
            train = df[(df["date"] >= "2000-01-01") & (df["date"] < f"{y}-01-01")]
            test = test_pool[test_pool["date"].dt.year == y]
            if len(train) < 100 or len(test) < 5:
                accs.append(np.nan); briers.append(np.nan); continue
            sw = train["t_low"].map(wf).values
            hm = make_pipe().fit(train[FEATS], train["home_score"],
                                 **{"p__sample_weight": sw})
            am = make_pipe().fit(train[FEATS], train["away_score"],
                                 **{"p__sample_weight": sw})
            r = eval_set(test, hm, am)
            accs.append(r["acc"]); briers.append(r["brier"])
        mean_acc = float(np.nanmean(accs))
        mean_b = float(np.nanmean(briers))
        accs_str = "  ".join(f"{a:.3f}" if not np.isnan(a) else " --- " for a in accs)
        print(f"{label:25s}  {accs_str}   {mean_acc:.3f}   {mean_b:.4f}")

    conn.close()


if __name__ == "__main__":
    main()
