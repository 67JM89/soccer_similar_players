"""
Phase 4b + 5: Train baseline model, backtest on 2022 WC, predict 2026 matchups.

Model:
  Poisson regression on each team's expected goals.
  Features: elo_home_pre, elo_away_pre, home_advantage,
            elo_diff (= elo_home + ha - elo_away)
  Target:   home_score and away_score (separate models)

Backtest:
  Train: matches 2000-01-01 .. 2021-12-31
  Test:  2022 FIFA World Cup matches (64 games)
  Metrics: accuracy (W/D/L), Brier score, log-loss

Predictor:
  Function predict_match(team_a, team_b, neutral=True) ->
    {expected goals, prob distributions, top match outcomes}
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
# Use elo_diff (which already includes home_advantage) + raw home_advantage flag
# Avoid redundant elo_home_pre/elo_away_pre to prevent collinearity
FEATURE_COLS = ["elo_diff", "home_advantage"]


def make_pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        ("poisson", PoissonRegressor(alpha=0.01, max_iter=1000)),
    ])


# ---------------------------------------------------------------- training
def load_features(conn) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM match_features", conn)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    return df


def split(df, train_end: str, test_tournament: str | None = None,
          test_year: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["date"] < train_end].copy()
    if test_tournament is not None:
        # Strict match: exact tournament name (no 'qualification', no 'women')
        mask = df["tournament"].str.lower() == test_tournament.lower()
        test = df[mask].copy()
        if test_year is not None:
            test = test[test["date"].dt.year == test_year]
    else:
        test = df[df["date"] >= train_end].copy()
    train = train[train["date"] >= "2000-01-01"]
    return train, test


def train_models(train: pd.DataFrame):
    Xh = train[FEATURE_COLS].values
    yh = train["home_score"].values
    ya = train["away_score"].values

    home_model = make_pipeline().fit(Xh, yh)
    away_model = make_pipeline().fit(Xh, ya)
    return home_model, away_model


# ---------------------------------------------------------------- prediction core
def outcome_probs(lam_h: float, lam_a: float, max_goals: int = 10) -> tuple[float, float, float]:
    """Given expected goals (Poisson rates), compute P(home win), P(draw), P(away win)."""
    ph = poisson.pmf(np.arange(max_goals + 1), lam_h)
    pa = poisson.pmf(np.arange(max_goals + 1), lam_a)
    grid = np.outer(ph, pa)
    p_home = np.tril(grid, -1).sum()
    p_draw = np.diag(grid).sum()
    p_away = np.triu(grid,  1).sum()
    return float(p_home), float(p_draw), float(p_away)


def predict_rows(home_m, away_m, df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].values
    lam_h = home_m.predict(X)
    lam_a = away_m.predict(X)
    probs = np.array([outcome_probs(h, a) for h, a in zip(lam_h, lam_a)])
    out = df.copy()
    out["lam_home"] = lam_h
    out["lam_away"] = lam_a
    out["p_home"]   = probs[:, 0]
    out["p_draw"]   = probs[:, 1]
    out["p_away"]   = probs[:, 2]
    out["pred_result"] = np.where(out["p_home"] > np.maximum(out["p_draw"], out["p_away"]), 1,
                          np.where(out["p_away"] > np.maximum(out["p_home"], out["p_draw"]), -1, 0))
    return out


# ---------------------------------------------------------------- evaluation
def evaluate(test: pd.DataFrame) -> dict:
    y_true = test["result"].values  # in {-1, 0, 1}
    # 3-class accuracy
    acc = accuracy_score(y_true, test["pred_result"])

    # Brier score (multiclass) — average over the three one-vs-rest Brier scores
    actual_h = (y_true == 1).astype(int)
    actual_d = (y_true == 0).astype(int)
    actual_a = (y_true == -1).astype(int)
    brier = (
        brier_score_loss(actual_h, test["p_home"]) +
        brier_score_loss(actual_d, test["p_draw"]) +
        brier_score_loss(actual_a, test["p_away"])
    ) / 3.0

    # Log loss
    P = test[["p_away", "p_draw", "p_home"]].values  # order: -1, 0, 1
    cls = (y_true + 1).astype(int)  # -1->0, 0->1, 1->2
    ll = log_loss(cls, P, labels=[0, 1, 2])

    return {"n": len(test), "accuracy": round(acc, 4),
            "brier": round(brier, 4), "log_loss": round(ll, 4)}


# ---------------------------------------------------------------- 2026 WC predictor
def make_predictor(conn, home_m, away_m):
    """Returns a function predict(team_a, team_b, neutral=True) using current ELO snapshot."""
    elo_df = pd.read_sql("SELECT team, elo FROM team_elo", conn)
    elo_lookup = dict(zip(elo_df["team"], elo_df["elo"]))

    # Use intl-results name if available
    aliases = pd.read_sql("SELECT canonical, intl_name FROM nation_aliases", conn)
    intl_map = dict(zip(aliases["canonical"], aliases["intl_name"]))

    BONUS = 100.0

    def lookup_elo(team: str) -> float:
        # try canonical first, then alias
        if team in elo_lookup: return elo_lookup[team]
        intl = intl_map.get(team, team)
        return elo_lookup.get(intl, np.nan)

    def predict(team_a: str, team_b: str, neutral: bool = True) -> dict:
        eh = lookup_elo(team_a)
        ea = lookup_elo(team_b)
        if np.isnan(eh) or np.isnan(ea):
            return {"error": f"ELO missing for {team_a} or {team_b}"}
        ha = 0 if neutral else BONUS
        # Match FEATURE_COLS order: elo_diff, home_advantage
        X = np.array([[(eh + ha) - ea, ha]])
        lam_h = float(home_m.predict(X)[0])
        lam_a = float(away_m.predict(X)[0])
        p_h, p_d, p_a = outcome_probs(lam_h, lam_a)
        return {
            "team_a": team_a, "team_b": team_b, "neutral": neutral,
            "elo_a": round(eh, 1), "elo_b": round(ea, 1),
            "expected_a_goals": round(lam_h, 2),
            "expected_b_goals": round(lam_a, 2),
            "p_a_win":  round(p_h, 3),
            "p_draw":   round(p_d, 3),
            "p_b_win":  round(p_a, 3),
        }
    return predict


# ---------------------------------------------------------------- main
def main():
    conn = sqlite3.connect(DB_PATH)
    df = load_features(conn)

    # ===== Backtest 1: 2022 World Cup =====
    print("="*60)
    print("Backtest: train < 2022-01-01, test = 2022 FIFA World Cup")
    print("="*60)
    train, test = split(df, "2022-11-01", "FIFA World Cup", 2022)
    print(f"Train: {len(train):,} matches  |  Test: {len(test)} matches")

    home_m, away_m = train_models(train)
    test_pred = predict_rows(home_m, away_m, test)
    print(f"\nMetrics on 2022 WC:  {evaluate(test_pred)}")

    print("\nFirst 10 predictions vs actual:")
    show = test_pred[["date","home_team","away_team","home_score","away_score",
                      "p_home","p_draw","p_away","pred_result"]].head(10)
    print(show.to_string(index=False))

    # ===== Backtest 2: 2018 World Cup =====
    print("\n" + "="*60)
    print("Backtest: train < 2018-01-01, test = 2018 FIFA World Cup")
    print("="*60)
    train2, test2 = split(df, "2018-06-01", "FIFA World Cup", 2018)
    print(f"Train: {len(train2):,} matches  |  Test: {len(test2)} matches")
    h2, a2 = train_models(train2)
    test_pred2 = predict_rows(h2, a2, test2)
    print(f"Metrics on 2018 WC:  {evaluate(test_pred2)}")

    # ===== Train final model on EVERYTHING for 2026 prediction =====
    print("\n" + "="*60)
    print("Final model: trained on ALL matches >= 2000-01-01")
    print("="*60)
    final_train = df[df["date"] >= "2000-01-01"]
    print(f"Train rows: {len(final_train):,}")
    final_h, final_a = train_models(final_train)
    h_pois = final_h.named_steps["poisson"]
    a_pois = final_a.named_steps["poisson"]
    print("Coefficients (home goals — on standardized features):")
    print(dict(zip(FEATURE_COLS, h_pois.coef_.round(5))))
    print(f"Intercept (home): {h_pois.intercept_:.4f}")
    print(f"Coefficients (away): {dict(zip(FEATURE_COLS, a_pois.coef_.round(5)))}")
    print(f"Intercept (away): {a_pois.intercept_:.4f}")

    # ===== Sample 2026 predictions =====
    print("\n" + "="*60)
    print("Sample 2026 predictions (neutral venue, USA host advantage NOT yet applied)")
    print("="*60)
    predict = make_predictor(conn, final_h, final_a)
    samples = [
        ("Spain", "Argentina"),
        ("France", "Brazil"),
        ("England", "Germany"),
        ("United States", "Mexico"),
        ("Japan", "South Korea"),
        ("Morocco", "Senegal"),
        ("Saudi Arabia", "Qatar"),
    ]
    for a, b in samples:
        r = predict(a, b, neutral=True)
        if "error" in r:
            print(f"  {a} vs {b}: {r['error']}")
            continue
        print(f"  {a} vs {b}:  "
              f"xG={r['expected_a_goals']}-{r['expected_b_goals']}  "
              f"P({a})={r['p_a_win']}  P(draw)={r['p_draw']}  P({b})={r['p_b_win']}")

    conn.close()


if __name__ == "__main__":
    main()
