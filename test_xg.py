"""
Phase 17 — Test: Does adding xG-top11 team-strength features improve v3.5?

Feature added:
  xg_diff = home_team's sum_xG_top11 (year-1) - away_team's sum_xG_top11 (year-1)

Walk-forward backtest on 2018, 2022 WCs (only years where xG data exists pre-match).
Compares v2.9 baseline (ELO + form) vs v2.9 + xg_diff.
"""
import sys, io, sqlite3
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

DB = Path(__file__).parent / "data" / "soccer.db"

BASE_FEATS = ["elo_diff", "home_advantage",
              "form_ppg_diff", "form_gd_diff", "form_cs_diff"]
XG_FEATS = BASE_FEATS + ["xg_diff"]


def make_pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        ("poisson", PoissonRegressor(alpha=0.01, max_iter=1000)),
    ])


def load_features() -> pd.DataFrame:
    conn = sqlite3.connect(DB)
    df = pd.read_sql("""
        SELECT m.*,
               f.h_form_ppg, f.h_form_gd, f.h_form_cs,
               f.a_form_ppg, f.a_form_gd, f.a_form_cs
        FROM match_features m
        JOIN match_form f ON f.date = m.date
            AND f.home_team = m.home_team AND f.away_team = m.away_team
    """, conn)
    df["date"] = pd.to_datetime(df["date"])
    df["form_ppg_diff"] = df["h_form_ppg"] - df["a_form_ppg"]
    df["form_gd_diff"]  = df["h_form_gd"]  - df["a_form_gd"]
    df["form_cs_diff"]  = df["h_form_cs"]  - df["a_form_cs"]

    # Join xG (use year-1 to avoid lookahead)
    xg = pd.read_sql("SELECT nation, year, sum_xG_top11 FROM team_xg_year", conn)
    df["match_year"] = df["date"].dt.year
    df["lookup_year"] = df["match_year"] - 1
    # Need to map team name → nation_aliases canonical
    al = pd.read_sql("SELECT canonical, fifa_name FROM nation_aliases", conn)
    # team in match_features is canonical; xg.nation is FIFA nationality string
    # Use alias to convert canonical → fifa_name for join
    alias_map = dict(zip(al["canonical"], al["fifa_name"]))
    df["home_fifa_nat"] = df["home_team"].map(alias_map).fillna(df["home_team"])
    df["away_fifa_nat"] = df["away_team"].map(alias_map).fillna(df["away_team"])

    # Build xG lookup keyed by (nation, year)
    xg_lookup = xg.set_index(["nation", "year"])["sum_xG_top11"]

    def lookup(nat_col, year_col):
        return df.apply(lambda r: xg_lookup.get((r[nat_col], r[year_col]), np.nan), axis=1)

    df["h_xg_top11"] = lookup("home_fifa_nat", "lookup_year")
    df["a_xg_top11"] = lookup("away_fifa_nat", "lookup_year")
    df["xg_diff"] = df["h_xg_top11"] - df["a_xg_top11"]
    conn.close()
    return df


def outcome_probs(lh, la, K=10):
    ph = poisson.pmf(np.arange(K + 1), lh)
    pa = poisson.pmf(np.arange(K + 1), la)
    g = np.outer(ph, pa)
    return float(np.tril(g, -1).sum()), float(np.diag(g).sum()), float(np.triu(g, 1).sum())


def evaluate(test, feats, hm, am):
    test = test.dropna(subset=feats).copy()
    if len(test) == 0:
        return {"n": 0, "accuracy": np.nan, "brier": np.nan, "log_loss": np.nan}
    X = test[feats].values
    lam_h = hm.predict(X); lam_a = am.predict(X)
    probs = np.array([outcome_probs(h, a) for h, a in zip(lam_h, lam_a)])
    test["p_home"] = probs[:, 0]; test["p_draw"] = probs[:, 1]; test["p_away"] = probs[:, 2]
    test["pred"] = np.where(test["p_home"] > np.maximum(test["p_draw"], test["p_away"]), 1,
                    np.where(test["p_away"] > np.maximum(test["p_home"], test["p_draw"]), -1, 0))
    y = test["result"].values
    return {
        "n": len(test),
        "accuracy": round(accuracy_score(y, test["pred"]), 4),
        "brier": round((
            brier_score_loss((y == 1).astype(int),  test["p_home"]) +
            brier_score_loss((y == 0).astype(int),  test["p_draw"]) +
            brier_score_loss((y == -1).astype(int), test["p_away"])
        ) / 3.0, 4),
        "log_loss": round(log_loss((y + 1).astype(int),
                                    test[["p_away","p_draw","p_home"]].values,
                                    labels=[0,1,2]), 4),
    }


MAJOR = {"fifa world cup", "uefa euro", "copa américa", "copa america",
         "africa cup of nations", "afc asian cup"}


def split(df, train_end, year):
    train = df[(df["date"] >= "2000-01-01") & (df["date"] < train_end)].copy()
    test = df[df["tournament"].str.lower().isin(MAJOR) & (df["date"].dt.year == year)].copy()
    return train, test


def fair_compare(df, year, train_end):
    """Train both models on the same training matches (those with xG features).
    Test both on the same test subset (those with xG features).
    Apples-to-apples comparison."""
    tr, te = split(df, train_end, year)

    # Train both models on the same xG-rich training set
    # (subset to rows with both BASE and xG features valid)
    tr_xg = tr.dropna(subset=XG_FEATS)
    te_xg = te.dropna(subset=XG_FEATS)

    print(f"\n### {year} major tournaments")
    print(f"  test_total_in_year={len(te)} | test_with_xg={len(te_xg)}")
    if len(te_xg) == 0 or len(tr_xg) < 100:
        print("  ⚠ insufficient data — skipping")
        return None

    # Baseline: v2.9 trained on same xG-rich rows but only uses BASE_FEATS
    Xb = tr_xg[BASE_FEATS].values
    hm_b = make_pipeline().fit(Xb, tr_xg["home_score"].values)
    am_b = make_pipeline().fit(Xb, tr_xg["away_score"].values)
    m_base = evaluate(te_xg, BASE_FEATS, hm_b, am_b)

    # xG-augmented
    Xx = tr_xg[XG_FEATS].values
    hm_x = make_pipeline().fit(Xx, tr_xg["home_score"].values)
    am_x = make_pipeline().fit(Xx, tr_xg["away_score"].values)
    m_xg = evaluate(te_xg, XG_FEATS, hm_x, am_x)

    print(f"  train={len(tr_xg):,}  test={len(te_xg)}")
    print(f"  v2.9 base : acc={m_base['accuracy']:.4f}  brier={m_base['brier']:.4f}  ll={m_base['log_loss']:.4f}")
    print(f"  v2.9 + xG : acc={m_xg['accuracy']:.4f}  brier={m_xg['brier']:.4f}  ll={m_xg['log_loss']:.4f}")
    d_acc = m_xg["accuracy"] - m_base["accuracy"]
    d_br  = m_xg["brier"]    - m_base["brier"]
    d_ll  = m_xg["log_loss"] - m_base["log_loss"]
    print(f"  Δ acc = {d_acc:+.4f} ({d_acc*100:+.2f}pp) | Δ brier = {d_br:+.4f} | Δ log_loss = {d_ll:+.4f}")
    return {"base": m_base, "xg": m_xg, "delta_acc": d_acc}


def main():
    df = load_features()
    print(f"Loaded {len(df):,} matches")
    n_with_xg = df.dropna(subset=XG_FEATS).shape[0]
    print(f"Rows with all xG features: {n_with_xg:,} ({100*n_with_xg/len(df):.1f}%)")
    print()
    print("=" * 70)
    print("FAIR COMPARISON: v2.9 vs v2.9+xG, same train/test subset")
    print("=" * 70)
    cases = [
        (2016, "2016-06-01"),   # Euro + Copa
        (2018, "2018-06-01"),   # WC
        (2019, "2019-06-01"),   # Copa
        (2021, "2021-06-01"),   # Euro
        (2022, "2022-11-01"),   # WC
        (2024, "2024-06-01"),   # Euro 2024 + Copa 2024
    ]
    results = []
    for year, te in cases:
        r = fair_compare(df, year, te)
        if r: results.append((year, r))

    # Aggregate
    if results:
        total_test = sum(r["base"]["n"] for _, r in results)
        weighted_base = sum(r["base"]["accuracy"] * r["base"]["n"] for _, r in results) / total_test
        weighted_xg   = sum(r["xg"]["accuracy"]   * r["xg"]["n"]   for _, r in results) / total_test
        print(f"\n{'='*70}")
        print(f"AGGREGATE ACROSS {len(results)} TOURNAMENTS ({total_test} matches)")
        print(f"  v2.9 base  : {weighted_base:.4f}")
        print(f"  v2.9 + xG  : {weighted_xg:.4f}")
        delta = weighted_xg - weighted_base
        # Standard error roughly 1.96 * sqrt(p(1-p)/n) for binary
        se = 1.96 * (weighted_base * (1 - weighted_base) / total_test) ** 0.5
        print(f"  Δ accuracy = {delta:+.4f} ({delta*100:+.2f}pp)  | 95% CI half-width ≈ ±{se*100:.2f}pp")
        if abs(delta) > se:
            verdict = "📈 SIGNIFICANT GAIN" if delta > 0 else "📉 SIGNIFICANT LOSS"
        else:
            verdict = "≈ within noise (not significant)"
        print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
